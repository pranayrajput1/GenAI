version_data_config = {
    'commit_message': 'test-commit',
    'image_tag': '0.0.1'
}

with open('cloudbuild.yaml', 'r') as template_file:
    template = template_file.read()

for key, value in version_data_config.items():
    template = template.replace(f'${{{key}}}', str(value))

with open('cloudbuild.yaml', 'w') as config_file:
    config_file.write(template)

# import requests
#
# # Replace these variables with your own values
# github_username = 'amanknoldus'
# github_repo = 'test-pipeline'
# github_token = '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'
#
# # Create the URL to fetch the latest commit
# url = f'https://api.github.com/repos/{github_username}/{github_repo}/commits?per_page=1'
#
# # Set the authorization header with your personal access token
# headers = {'Authorization': f'token {github_token}'}
#
# # Make the API request
# response = requests.get(url, headers=headers)
#
# if response.status_code == 200:
#     commit_data = response.json()
#     if len(commit_data) > 0:
#         latest_commit_message = commit_data[0]['commit']['message']
#         print(f'Latest commit message: {latest_commit_message}')
#     else:
#         print('No commits found in the repository.')
# else:
#     print(f'Failed to fetch the latest commit. Status code: {response.status_code}')

  # - name: 'ubuntu'
  #   entrypoint: 'bash'
  #   args:
  #   - -c
  #   - |
  #     apt-get update
  #     apt-get install -y git
  #     git_log_output=$(git log -1 --pretty=format:"%s")
  #   env:
  #   - 'COMMIT_MESSAGE=$git_log_output'