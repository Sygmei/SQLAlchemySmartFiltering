from datetime import date, datetime, time
from decimal import Decimal
import re
from typing import Any, Callable, TypedDict

from sqlalchemy import (
    Boolean,
    ColumnElement,
    ColumnExpressionArgument,
    Date,
    DateTime,
    Float,
    FromClause,
    Integer,
    Numeric,
    Select,
    String,
    Time,
    or_,
)
from sqlalchemy import cast as sqlcast
from sqlalchemy import inspect as sqlinspect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql.selectable import Join
from sqlalchemy.sql.type_api import TypeEngine


FieldModifier = Callable[[ColumnElement[Any]], ColumnElement[Any]]
OrmFilterTarget = type[DeclarativeBase]
FilterTarget = FromClause | OrmFilterTarget


class ParsedFilter(TypedDict, total=False):
    original_filter: str
    field: str
    op: str
    value: str | None
    field_modifier: FieldModifier


class InvalidFilteringColumn(Exception):
    def __init__(
        self,
        filter_query: str,
        target_type: str,
        column: str,
        available_columns: list[str],
    ):
        self.filter_query = filter_query
        self.target_type = target_type
        self.column = column
        self.available_columns = available_columns

    def __str__(self):
        return (
            f"Invalid filtering column '{self.column}' in filter query '{self.filter_query}' "
            f"for target type '{self.target_type}'. Available columns are: {', '.join(self.available_columns)}."
        )


class InvalidFilterOperator(Exception):
    def __init__(self, filter_query: str, operator: str):
        self.filter_query = filter_query
        self.operator = operator

    def __str__(self):
        return f"Invalid filter operator '{self.operator}' in filter query '{self.filter_query}'."


class AmbiguousFilterTarget(Exception):
    def __str__(self):
        return (
            "Cannot infer a primary filtering target from this query. "
            "Pass table_hint to indicate which table should receive unqualified filters."
        )


class InvalidFilterValue(Exception):
    def __init__(self, filter_query: str, column: str, value: str, target_type: str):
        self.filter_query = filter_query
        self.column = column
        self.value = value
        self.target_type = target_type

    def __str__(self):
        return (
            f"Invalid filter value '{self.value}' for column '{self.column}' "
            f"of type '{self.target_type}' in filter query '{self.filter_query}'."
        )

SYMBOL_OPERATORS = {
    "=": "eq",
    "==": "eq",
    "!=": "ne",
    ">": "gt",
    "<": "lt",
    ">=": "ge",
    "<=": "le",
    "~=": "like",
}
OPERATORS_ACCESSORS = {
    "eq": lambda c: c.__eq__,
    "ne": lambda c: c.__ne__,
    "gt": lambda c: c.__gt__,
    "lt": lambda c: c.__lt__,
    "ge": lambda c: c.__ge__,
    "le": lambda c: c.__le__,
    "in": lambda c: c.in_,
    "not_in": lambda c: c.not_in,
    "like": lambda c: c.like,
}
VERB_OPERATORS = set(OPERATORS_ACCESSORS.keys())
NULL_SYMBOL = "<null>"

SYMBOL_OPERATOR_PATTERN = "|".join(
    re.escape(operator) for operator in sorted(SYMBOL_OPERATORS, key=len, reverse=True)
)
VERB_OPERATOR_PATTERN = "|".join(
    re.escape(operator) for operator in sorted(VERB_OPERATORS, key=len, reverse=True)
)
GENERAL_FILTER_SHAPE = re.compile(
    r"^(?P<field>[a-zA-Z._]+)"
    rf"(?P<op>\s+(?:{VERB_OPERATOR_PATTERN})\s+|\s*(?:{SYMBOL_OPERATOR_PATTERN})\s*)"
    r"(?P<value>.*)?$"
)


def split_filter_spec(filter_spec: str) -> list[str]:
    filter_parts: list[str] = []
    current_filter_part: list[str] = []
    active_quote: str | None = None
    escaped = False

    for char in filter_spec:
        if escaped:
            if char in {active_quote, "\\"}:
                current_filter_part.append(char)
            else:
                current_filter_part.extend(["\\", char])
            escaped = False
            continue
        if char == "\\" and active_quote is not None:
            escaped = True
            continue
        if char in {'"', "'"}:
            if active_quote is None:
                active_quote = char
            elif active_quote == char:
                active_quote = None
        if char == "," and active_quote is None:
            filter_parts.append("".join(current_filter_part))
            current_filter_part = []
            continue
        current_filter_part.append(char)

    if escaped:
        current_filter_part.append("\\")
    filter_parts.append("".join(current_filter_part))
    return filter_parts


def is_quoted_filter_value(value: str) -> bool:
    return len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}


def unquote_filter_value(value: str) -> str:
    if not is_quoted_filter_value(value):
        return value
    quote = value[0]
    return value[1:-1].replace(f"\\{quote}", quote).replace("\\\\", "\\")


def normalize_datetime_value(value: str) -> str:
    if value.endswith("Z"):
        return value[:-1] + "+00:00"
    return value


def parse_bool_value(value: str) -> bool:
    normalized_value = value.lower()
    if normalized_value in {"true", "1", "yes"}:
        return True
    if normalized_value in {"false", "0", "no"}:
        return False
    raise ValueError


def coerce_filter_value(
    value: str,
    column_type: TypeEngine[Any],
) -> Any:
    if isinstance(column_type, DateTime):
        return datetime.fromisoformat(normalize_datetime_value(value))
    if isinstance(column_type, Date):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return datetime.fromisoformat(normalize_datetime_value(value)).date()
    if isinstance(column_type, Time):
        return time.fromisoformat(normalize_datetime_value(value))
    if isinstance(column_type, Integer):
        return int(value)
    if isinstance(column_type, Float):
        return float(value)
    if isinstance(column_type, Numeric):
        return Decimal(value)
    if isinstance(column_type, Boolean):
        return parse_bool_value(value)
    return value


def coerce_filter_value_for_column(
    value: str | list[str],
    column: Any,
    parsed_single_filter: ParsedFilter,
) -> Any:
    column_type = column.type
    try:
        if isinstance(value, list):
            return [
                coerce_filter_value(unquote_filter_value(item.strip()), column_type)
                for item in value
            ]
        return coerce_filter_value(value, column_type)
    except ValueError as error:
        raise InvalidFilterValue(
            filter_query=parsed_single_filter["original_filter"],
            column=parsed_single_filter["field"],
            value=str(value),
            target_type=str(column_type),
        ) from error


def get_orm_mapper(target: Any) -> Any | None:
    inspected = sqlinspect(target, raiseerr=False)
    if inspected is None or not hasattr(inspected, "relationships"):
        return None
    return inspected


def is_orm_target(target: Any) -> bool:
    return get_orm_mapper(target) is not None


def describe_filter_target(target: FilterTarget) -> str:
    mapper = get_orm_mapper(target)
    if mapper is not None:
        return mapper.class_.__name__
    return str(getattr(infer_table(target), "description", target))


def list_filter_target_fields(target: FilterTarget) -> list[str]:
    mapper = get_orm_mapper(target)
    if mapper is not None:
        return list(mapper.column_attrs.keys()) + list(mapper.relationships.keys())
    return [column.key for column in reversed(list(target.columns))]


def infer_table(table_or_join: FromClause) -> FromClause:
    if isinstance(table_or_join, Join):
        return table_or_join.left
    return table_or_join


def infer_orm_filter_target(query: Select) -> OrmFilterTarget | None:
    entities: list[OrmFilterTarget] = []
    for description in query.column_descriptions:
        entity = description.get("entity")
        if entity is not None and is_orm_target(entity) and entity not in entities:
            entities.append(entity)
    if len(entities) == 1:
        return entities[0]
    return None


def infer_filter_target(query: Select) -> FilterTarget:
    orm_target = infer_orm_filter_target(query)
    if orm_target is not None:
        return orm_target

    final_froms = query.get_final_froms()
    if len(final_froms) != 1 or isinstance(final_froms[0], Join):
        raise AmbiguousFilterTarget()
    return final_froms[0]


def rebuild_incomplete_filter(
    target: FilterTarget,
    single_filter: str,
) -> list[ParsedFilter]:
    mapper = get_orm_mapper(target)
    autosearch_source = mapper.local_table if mapper is not None else infer_table(target)
    autosearch_fields = dict(getattr(autosearch_source, "kwargs", {})).get(
        "autosearch_fields",
        {},
    )
    return [
        {
            "field": field,
            "op": "like",
            "value": f"%{single_filter}%",
            "original_filter": single_filter,
            "field_modifier": lambda c: sqlcast(c, String),
        }
        for field in autosearch_fields
    ]


def parse_single_filter(single_filter: str) -> list[ParsedFilter] | None:
    res = GENERAL_FILTER_SHAPE.match(single_filter)
    if res:
        operator = res.group("op")
        if operator.strip() not in VERB_OPERATORS | SYMBOL_OPERATORS.keys():
            raise InvalidFilterOperator(single_filter, operator)
        return [
            {
                "original_filter": single_filter,
                "field": res.group("field"),
                "op": operator,
                "value": res.group("value"),
            }
        ]
    return None


def parse_filter_spec(
    target: FilterTarget,
    filter_spec: str,
) -> list[list[ParsedFilter]]:
    filter_parts = split_filter_spec(filter_spec)
    parsed_filter_parts = []
    for filter_part in filter_parts:
        parsed_filter_part = parse_single_filter(single_filter=filter_part)
        if parsed_filter_part:
            parsed_filter_parts.append(parsed_filter_part)
        else:
            parsed_filter_parts.append(
                rebuild_incomplete_filter(target=target, single_filter=filter_part)
            )
    return parsed_filter_parts


def build_column_filter(
    column: Any,
    parsed_single_filter: ParsedFilter,
) -> ColumnExpressionArgument[bool]:
    if "field_modifier" in parsed_single_filter:
        column = parsed_single_filter["field_modifier"](column)
    operator = parsed_single_filter["op"].strip()
    if operator in SYMBOL_OPERATORS:
        operator = SYMBOL_OPERATORS[operator]
    raw_column_value = (parsed_single_filter.get("value") or "").strip()
    is_quoted_value = is_quoted_filter_value(raw_column_value)
    raw_column_value = unquote_filter_value(raw_column_value)
    if (
        not is_quoted_value
        and raw_column_value == NULL_SYMBOL
        and operator in {"eq", "ne"}
    ):
        return column.is_(None) if operator == "eq" else column.is_not(None)
    column_operator = OPERATORS_ACCESSORS[operator](column)
    column_value: str | list[str] = raw_column_value
    if (
        not is_quoted_value
        and raw_column_value.startswith("[")
        and raw_column_value.endswith("]")
    ):
        column_value = [
            column_value_item.strip() for column_value_item in raw_column_value.split("|")
        ]
        if column_value:
            column_value[0] = column_value[0].removeprefix("[")
            column_value[-1] = column_value[-1].removesuffix("]")
    if operator != "like":
        column_value = coerce_filter_value_for_column(
            value=column_value,
            column=column,
            parsed_single_filter=parsed_single_filter,
        )
    return column_operator(column_value)


def build_table_filter(
    table: FromClause,
    parsed_single_filter: ParsedFilter,
) -> ColumnExpressionArgument[bool]:
    columns_map = {column.key: column for column in reversed(list(table.columns))}
    single_filter_column = parsed_single_filter["field"]
    if single_filter_column not in columns_map:
        raise InvalidFilteringColumn(
            filter_query=parsed_single_filter["original_filter"],
            target_type=describe_filter_target(table),
            column=single_filter_column,
            available_columns=list(columns_map.keys()),
        )
    return build_column_filter(
        column=columns_map[single_filter_column],
        parsed_single_filter=parsed_single_filter,
    )


def build_orm_path_filter(
    target: OrmFilterTarget,
    field_path: list[str],
    parsed_single_filter: ParsedFilter,
) -> ColumnExpressionArgument[bool]:
    mapper = get_orm_mapper(target)
    if mapper is None:
        raise InvalidFilteringColumn(
            filter_query=parsed_single_filter["original_filter"],
            target_type=str(target),
            column=parsed_single_filter["field"],
            available_columns=[],
        )

    field_name = field_path[0]
    if len(field_path) == 1:
        if field_name not in mapper.column_attrs:
            raise InvalidFilteringColumn(
                filter_query=parsed_single_filter["original_filter"],
                target_type=describe_filter_target(target),
                column=parsed_single_filter["field"],
                available_columns=list_filter_target_fields(target),
            )
        return build_column_filter(
            column=getattr(target, field_name),
            parsed_single_filter=parsed_single_filter,
        )

    if field_name not in mapper.relationships:
        raise InvalidFilteringColumn(
            filter_query=parsed_single_filter["original_filter"],
            target_type=describe_filter_target(target),
            column=parsed_single_filter["field"],
            available_columns=list_filter_target_fields(target),
        )

    relationship = mapper.relationships[field_name]
    relationship_filter = build_orm_path_filter(
        target=relationship.mapper.class_,
        field_path=field_path[1:],
        parsed_single_filter=parsed_single_filter,
    )
    relationship_attribute = getattr(target, field_name)
    if relationship.uselist:
        return relationship_attribute.any(relationship_filter)
    return relationship_attribute.has(relationship_filter)


def build_orm_filter(
    target: FilterTarget,
    parsed_single_filter: ParsedFilter,
) -> ColumnExpressionArgument[bool]:
    mapper = get_orm_mapper(target)
    if mapper is not None:
        return build_orm_path_filter(
            target=mapper.class_,
            field_path=parsed_single_filter["field"].split("."),
            parsed_single_filter=parsed_single_filter,
        )
    return build_table_filter(
        table=infer_table(target),
        parsed_single_filter=parsed_single_filter,
    )


def apply_filters_from_filter_spec(
    query: Select, filter_spec: str, table_hint: FilterTarget | None = None
) -> Select:
    final_query: Select = query
    target: FilterTarget = (
        infer_table(table_hint)
        if isinstance(table_hint, FromClause)
        else table_hint
        if table_hint is not None
        else infer_filter_target(query)
    )

    parsed_filter_spec = parse_filter_spec(target=target, filter_spec=filter_spec)

    for single_filter in parsed_filter_spec:
        orm_filters = [
            build_orm_filter(
                target=target,
                parsed_single_filter=single_filter_part,
            )
            for single_filter_part in single_filter
        ]
        final_query = final_query.where(or_(*orm_filters))
    return final_query
