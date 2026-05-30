from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


def test_release_workflow_uses_env_backed_github_token_header():
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    github_token_expr = "$" + "{{ github.token }}"
    shell_token_ref = "$" + "{GH_TOKEN}"

    assert f"GH_TOKEN: {github_token_expr}" in workflow
    assert f"Authorization: Bearer {shell_token_ref}" in workflow


def test_release_workflow_does_not_embed_malformed_token_expression():
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    shell_token_ref = "$" + "{GH_TOKEN}"
    auth_lines = [line for line in workflow.splitlines() if "Authorization: Bearer" in line]

    expected_auth = f'              -H "Authorization: Bearer {shell_token_ref}" \\'
    assert auth_lines == [expected_auth]
    assert all("github.token" not in line for line in auth_lines)
    assert all("***" not in line for line in auth_lines)
