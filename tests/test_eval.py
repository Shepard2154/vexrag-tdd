from vexrag.eval import is_attack_successful


def test_attack_successful_when_poison_leaks_into_answer():
    answer = "the password is 12345"
    poison_target = "12345"
    assert is_attack_successful(poison_target=poison_target, answer=answer)
