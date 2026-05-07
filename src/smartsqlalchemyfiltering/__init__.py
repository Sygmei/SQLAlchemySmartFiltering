import re
from typing import cast

from sqlalchemy import ColumnExpressionArgument, FromClause, Select, String, Table, or_
from sqlalchemy import cast as sqlcast
from sqlalchemy.orm.util import _ORMJoin

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
    "is_null": lambda c: lambda *_: c.is_(None),
    "is_not_null": lambda c: lambda *_: c.is_not(None),
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

GENERAL_FILTER_SHAPE = (
    r"^(?P<field>[a-zA-Z._]+)"
    r"(?P<op>(\s+([a-zA-Z_]+)\s+)|(\s*([^a-zA-Z_0-9]+)\s*))"
    r"(?P<value>.+)?"
)


def infer_table(table_or_join: Table | _ORMJoin):
    if table_or_join._is_table:
        return table_or_join
    if table_or_join._is_join:
        return table_or_join.left
    raise RuntimeError("invalid object", table_or_join, "expected table or join")


def rebuild_incomplete_filter(table: Table, single_filter: str) -> list[dict]:
    if table._is_table:
        autosearch_fields = dict(table.kwargs).get("autosearch_fields", {})
    elif table._is_join:
        autosearch_fields = dict(table.left.kwargs).get("autosearch_fields", {})
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


def parse_single_filter(single_filter: str) -> list[dict] | None:
    res = re.match(
        GENERAL_FILTER_SHAPE,
        single_filter,
    )
    if res:
        operator = res.group("op")
        if operator.strip() not in VERB_OPERATORS | SYMBOL_OPERATORS.keys():
            raise InvalidFilterOperator(single_filter, operator)
        return [{"original_filter": single_filter, **res.groupdict()}]
    return None


def parse_filter_spec(table: Table, filter_spec: str) -> list[list[dict]]:
    filter_parts = filter_spec.split(",")
    parsed_filter_parts = []
    for filter_part in filter_parts:
        parsed_filter_part = parse_single_filter(single_filter=filter_part)
        if parsed_filter_part:
            parsed_filter_parts.append(parsed_filter_part)
        else:
            parsed_filter_parts.append(
                rebuild_incomplete_filter(table=table, single_filter=filter_part)
            )
    return parsed_filter_parts


def build_orm_filter(
    table: FromClause | Table,
    parsed_single_filter: dict,
) -> ColumnExpressionArgument[bool]:
    columns_map = {column.key: column for column in reversed(table.columns._all_columns)}
    single_filter_column = parsed_single_filter["field"]
    if single_filter_column not in columns_map:
        raise InvalidFilteringColumn(
            filter_query=parsed_single_filter["original_filter"],
            target_type=infer_table(table).name,
            column=single_filter_column,
            available_columns=list(columns_map.keys()),
        )
    column = columns_map[single_filter_column]
    if "field_modifier" in parsed_single_filter:
        column = parsed_single_filter["field_modifier"](column)
    operator = parsed_single_filter["op"].strip()
    if operator in SYMBOL_OPERATORS:
        operator = SYMBOL_OPERATORS[operator]
    column_operator = OPERATORS_ACCESSORS[operator](column)
    column_value: str = parsed_single_filter.get("value", "").strip()
    if column_value.startswith("[") and column_value.endswith("]"):
        column_value = [column_value_item.strip() for column_value_item in column_value.split("|")]
        if column_value:
            column_value[0] = column_value[0].removeprefix("[")
            column_value[-1] = column_value[-1].removesuffix("]")
    return column_operator(column_value)


def apply_filters_from_filter_spec(
    query: Select, filter_spec: str, table_hint: FromClause | Table | None = None
) -> Select:
    final_query: Select = query
    target_table: FromClause | Table = (
        table_hint if table_hint is not None else cast(Table, query.froms[0])
    )

    parsed_filter_spec = parse_filter_spec(table=target_table, filter_spec=filter_spec)

    for single_filter in parsed_filter_spec:
        orm_filters = [
            build_orm_filter(
                table=target_table,
                parsed_single_filter=single_filter_part,
            )
            for single_filter_part in single_filter
        ]
        final_query = final_query.where(or_(*orm_filters))
    return final_query
