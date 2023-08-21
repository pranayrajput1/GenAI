from google.cloud import aiplatform


def undeploy_model_from_endpoint(names):
    endpoints = aiplatform.Endpoint.list()
    for i in endpoints:
        if str(i.display_name) == names:
            i.undeploy_all()


undeploy_model_from_endpoint("dolly_v2_3b_endpoint")
