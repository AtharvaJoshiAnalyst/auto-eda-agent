"""Layer 1 security tests: dangerous code is refused before it ever runs."""

import pytest

from agent.code_check import check_code


@pytest.mark.parametrize("code", [
    "result = df.groupby('region')['revenue'].sum()",
    "import plotly.express as px\nresult = px.bar(df, x='region', y='revenue')",
    "result = df['revenue'].apply(lambda v: v * 2).max()",
    "result = df.to_dict()",                                            # safe 'to_' method
])
def test_normal_analysis_is_allowed(code):
    assert check_code(code).ok


@pytest.mark.parametrize("attack", [
    "import os\nresult = os.listdir('/')",                              # other modules
    "import subprocess\nresult = subprocess.run(['ls'])",
    "result = open('/etc/passwd').read()",                              # files
    "result = eval('1+1')",                                             # running strings as code
    "result = __import__('os').getcwd()",                               # sneaky import
    "result = ().__class__.__bases__[0].__subclasses__()",              # classic Python escape
    "result = getattr(df, 'to_csv')('stolen.csv')",                     # hiding a name in a string
    "df.to_csv('stolen.csv')\nresult = 1",                              # writing files
    "result = pd.read_csv('https://example.com/data.csv')",             # downloading
    "result = pd.io.common.urlopen('https://example.com')",             # module internals
    "result = np.load('/tmp/x.npy')",
])
def test_attacks_are_blocked(attack):
    result = check_code(attack)
    assert not result.ok
    assert result.problems  # every refusal explains why


def test_code_must_set_result():
    result = check_code("x = df['revenue'].sum()")
    assert not result.ok and "result" in result.problems[0]


def test_syntax_errors_are_reported_not_raised():
    result = check_code("result = (")
    assert not result.ok and "Syntax error" in result.problems[0]