# context_cleaner.py
#清洗memory，只将memory中有用的部分提纯

def clean_memory_for_skill(memory: dict, skill: dict) -> dict:
    skill_name = skill["name"]

    if skill_name == "experiment_design_skill":
        allowed_keys = [
            "project_title",
            "project_type",
            "main_goal",
            "research_focus",
            "possible_datasets",
            "possible_metrics",
            "current_user_background",
        ]

    elif skill_name in {
        "paper_analysis_skill",
        "paper_search_skill",
        "paper_import_skill",
    }:
        allowed_keys = [
            "project_title",
            "main_goal",
            "research_focus",
            "possible_datasets",
            "current_user_background",
        ]

    elif skill_name == "teacher_student_model_skill":
        allowed_keys = [
            "project_title",
            "main_goal",
            "research_focus",
            "current_user_background",
        ]

    elif skill_name == "dataset_selection_skill":
        allowed_keys = [
            "project_title",
            "main_goal",
            "possible_datasets",
            "possible_metrics",
            "current_user_background",
        ]

    elif skill_name == "email_drafting_skill":
        allowed_keys = [
            "project_title",
            "project_type",
            "main_goal",
            "research_focus",
            "possible_datasets",
            "possible_metrics",
            "current_user_background",
        ]

    elif skill_name == "file_analysis_skill":
        # File questions should be grounded in the file first. Candidate
        # datasets and metrics are added later only when the task needs them.
        allowed_keys = [
            "project_title",
            "main_goal",
            "research_focus",
        ]
    else:
        allowed_keys = [
            "project_title",
            "main_goal",
            "research_focus",
            "current_user_background",
        ]

    cleaned_memory = {}

    for key in allowed_keys:
        if key in memory:
            cleaned_memory[key] = memory[key]

    return cleaned_memory
