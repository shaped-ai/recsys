from dataclasses import dataclass
from typing import List


@dataclass
class feature:
    name: str
    dtype: str
    unique_value_count: int


def dataframe_schema(df) -> List[feature]:
    r = []
    for col in df.columns.values:
        col_feature = feature(
            name=col, dtype=df[col].dtype.name, unique_value_count=len(df[col].unique())
        )
        r.append(col_feature)

    return r
