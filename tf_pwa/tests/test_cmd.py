import subprocess
from tf_pwa.main import regist_subcommand, main


@regist_subcommand()
def sss(i: int, *args, j="ss"):
    print(i, type(i))
    return i


def test_cmd():
    ret = subprocess.getoutput("python -m tf_pwa")
    

def test_main_f():
    ret = main(["sss", "2", "--j=dd"])
    assert ret == 2
