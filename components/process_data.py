from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies


# @component(
#     base_image="python:3.10.6",
#     packages_to_install=resolve_dependencies(
#         'pandas',
#         'google-cloud-storage',
#         'gcsfs',
#         'pyarrow',
#         'fsspec'
#     )
# )
def process_data(
        # dataset: dsl.Output[dsl.Dataset]
):
    import logging
    import pandas as pd

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        logging.debug("Task: Getting dataset from gcs bucket: 'llm-bucket-dolly'")
        read_file = pd.read_csv("gs://llm-bucket-dolly/query_train.csv")
        train_df = pd.DataFrame(read_file)

        logging.debug("Task: Saving dataset to parquet file")
        # train_df.to_parquet(dataset.path)

    except Exception as e:
        logging.error("Failed to process data!")
        raise e


process_data()
