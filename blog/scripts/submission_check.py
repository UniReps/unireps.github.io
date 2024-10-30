# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gitpython",
# ]
# ///
import os
from pathlib import Path
import re
import sys
import git

_name_format_re = re.compile(r"(\d{4}-\d{2}-\d{2})-(\w+)")

_ALLOWED_DIRS = {"_posts", "assets"}


def is_new_file(repo: git.Repo, file_path: Path, base_branch: str):
    """
    Check if a file is new in the PR by comparing the current branch to the base branch.
    A file is new if it doesn't exist in the base branch.
    """
    try:
        # Try to retrieve the file from the base branch
        show_result = repo.git.show(f"{base_branch}:{file_path}")
        print(f"{show_result=}")
        # If the file exists in the base branch, it's not new
        return False
    except git.exc.GitCommandError:
        # If the file does not exist in the base branch, it's new
        return True


def check_submission(base_branch: str):
    """Check if the submission meets the required guidelines."""

    # Initialize Git repository object
    repo = git.Repo(search_parent_directories=True)

    # Compare PR branch changes with the base branch
    diff = repo.git.diff(f"origin/{base_branch}..HEAD", name_only=True)
    # diff = repo.git.diff(base_branch, "--staged", name_only=True)

    # Get the list of changed files
    changed_files = [Path(file) for file in diff.splitlines()]
    print(f"Changed files: {changed_files}")

    # Ensure that all the changes are confined to the assets and _posts directory
    for file in changed_files:
        if file.parts[0] not in _ALLOWED_DIRS:
            raise Exception(
                f"All changes must be confined to the {_ALLOWED_DIRS} directories. Found: {file.parts}"
            )

    for file in changed_files:
        print(f"Checking file: {file}")
        if not is_new_file(repo, file, base_branch):
            raise Exception(
                f"Only new files are allowed in the submission/PR. Found a not allowed modification to: {file}"
            )

    # Get the list of changed files
    post_files = [
        file
        for file in changed_files
        if file.parts[0] == "_posts" and file.suffix == ".md"
    ]

    # Ensure only one file is added in _posts
    if len(post_files) != 1:
        raise Exception(
            f"Exactly one .md file should be added in the _posts directory. Found files: {post_files}"
        )

    post_file = post_files[0]

    re_match = _name_format_re.match(post_file.stem)

    # Check post file naming format
    if not re_match:
        raise Exception(
            f"The post file must follow the naming convention YYYY-MM-DD-title.md. Found: {post_file.name}"
        )

    submission_key = re_match.group(0)
    # _submission_date = re_match.group(1)
    # _submission_name = re_match.group(2)

    with open(os.environ["GITHUB_ENV"], "a") as env_file:
        env_file.write(f"SUBMISSION_KEY={submission_key}\n")

    # Define paths for the directories and bib file
    asset_files = [file for file in changed_files if file.parts[0] == "assets"]

    if asset_files:
        # If there are changes in the assets directory, ensure they are confined to the right folders
        for file in asset_files:
            try:
                submission_key_index = file.parts.index(submission_key)
                if submission_key_index < 2:
                    raise ValueError
            except ValueError:
                raise Exception(
                    f"All asset files must be confined to a submission-specific directory. E.g. assets/asset_type_dir/{submission_key}/*. Found: {file}"
                )

            if "bibliography" in file.parts:
                if not file.suffix == ".bib":
                    raise Exception(
                        f"Only .bib files are allowed in the bibliography directory. Found: {file}"
                    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python submission_check.py <base_branch>")

    base_branch = sys.argv[1]

    try:
        check_submission(base_branch=base_branch)

    except Exception as e:
        # Catch any exceptions and report them in the GitHub Actions format
        print(f"::error::{str(e)}")
        sys.exit(1)

    print("Submission check passed!")
