from typing import List

DEPENDENCY_VERSIONS = {
    "numpy": "numpy==1.23.5",
    "pandas": "pandas==2.0.2",
    "scikit-learn": "scikit-learn==1.2.2",
    "fsspec": "fsspec==2023.3.0",
    "pyarrow": "pyarrow==9.0.0",
    "gcsfs": "gcsfs==2023.3.0",
    "google-cloud-storage": "google-cloud-storage==2.9.0",
    "google-cloud-aiplatform": "google-cloud-aiplatform==1.26.0",
    "google-cloud-build": "google-cloud-build==3.16.0",
    "kfp": "kfp[all]==1.8.19",
    "matplotlib": "matplotlib==3.7.1"
}


def resolve_dependencies(*names) -> List[str]:
    return [DEPENDENCY_VERSIONS[name] for name in names]
