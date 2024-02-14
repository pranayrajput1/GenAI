string_template = """
{TASK}
{RESUME}
{DECORATOR}
Response: """

reference = """Here's the structured resume:
Name: Name of Candidate
Email: email
Mobile: mobile number

Technical Skills: Technical Skills present in the resume.
    
Education: Education Qualification present in the resume

Other Information:

* Email: candidate email
* Mobile No: mobile number

Experience:

* Job title, Organisation Name, Location (Start Date - End Date)
+ Experience mentioned in the resume.	
"""

task = "Extract all the skills present in the below mentioned fresh resume and store it in python list format"

decorator = "Remove all unnecessary information if present in the response"
