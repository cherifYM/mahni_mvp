def extract_skills(text):
    """
    Extracts skills from the given text.

    Args:
        text (str): The input text from which skills need to be extracted.

    Returns:
        list: A list of extracted skills found in the text.
    """
    predefined_skills = [
        "Python", "Java", "SQL", "Machine Learning", "Data Analysis",
        "Communication", "Leadership", "Project Management"
    ]
    extracted_skills = [skill for skill in predefined_skills if skill.lower() in text.lower()]
    return extracted_skills