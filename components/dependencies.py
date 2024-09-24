from typing import List

DEPENDENCY_VERSIONS = {
    "pandas": "pandas==2.0.2",
    "fsspec": "fsspec==2023.3.0",
    "pyarrow": "pyarrow==9.0.0",
    "gcsfs": "gcsfs==2023.3.0",
    "google-cloud-storage": "google-cloud-storage==2.9.0",
    "google-cloud-aiplatform": "google-cloud-aiplatform[all]==1.67.1",
    "google-cloud-bigquery": "google-cloud-bigquery==3.25.0",
    "google-cloud-build": "google-cloud-build==3.16.0",
    "kfp": "kfp[all]==1.8.19",
}


def resolve_dependencies(*names) -> List[str]:
    return [DEPENDENCY_VERSIONS[name] for name in names]
