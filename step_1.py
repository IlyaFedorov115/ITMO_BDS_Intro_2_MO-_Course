def solve(a, b):
    if a == b == 0:
        return 'Any'
    elif a == 0 and b != 0:
        return 'Error'
    else:
        return b / a





def grade(score):
    if score >= 0 and score < 60:
        return "неудовлетворительно"
    elif score >= 60 and score <= 74:
        return "удовлетворительно"
    elif score > 74 and score <= 90:
        return "хорошо"
    elif score > 90:
        return "отлично"
