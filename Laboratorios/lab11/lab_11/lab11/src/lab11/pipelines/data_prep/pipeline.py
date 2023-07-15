"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    create_model_input_table,
    get_data,
    preprocess_companies,
    preprocess_shuttles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                get_data,
                inputs=None,
                outputs=["companies", "shuttles", "reviews"],
                name="get_data",
            ),
            node(
                preprocess_companies,
                inputs="companies",
                outputs="preprocessed_companies",
                name="preprocess_companies",
            ),
            node(
                preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="preprocess_shuttles",
            ),
            node(
                create_model_input_table,
                inputs=dict(
                    shuttles="preprocessed_shuttles",
                    companies="preprocessed_companies",
                    reviews="reviews",
                ),
                outputs="model_input_table",
                name="create_model_input_table",
            ),
        ]
    )
