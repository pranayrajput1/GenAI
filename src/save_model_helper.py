def save_model(bucket_name: str, model_dir):
    """
    Function to save model files to the gcs_bucket.
    @param bucket_name: gcs_bucket name where model files has to be saved
    @param model_dir:local model files directory
    """
    import logging
    from google.cloud import storage
    import os

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        logging.debug("Task: Initializing GCP Storage Client")
        client = storage.Client()

        logging.debug("Task: Getting Bucket")
        bucket = client.get_bucket(bucket_name)

        logging.debug(f"Task: Listing files present in the directory: {model_dir}")
        files = os.listdir(model_dir)

        logging.debug(f"Task: Iterating over each file present in the directory: {model_dir}")
        for file_name in files:
            file_path = os.path.join(model_dir, file_name)

            if os.path.isfile(file_path):
                logging.debug(f"Task: Creating blob object to upload the file to GCS Bucket: {bucket_name}")
                blob = bucket.blob(file_name)
                blob.upload_from_filename(file_path)

                logging.debug(f"Task: Uploaded file: {file_name} to {bucket_name} successfully")

    except Exception as e:
        logging.error(f"Some error occurred in uploading model files to the bucket nameL: {bucket_name}")
        raise e
