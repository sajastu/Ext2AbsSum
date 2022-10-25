import re


def remove_et_als(text):

    text = re.sub(r"\(\w*?\)", r"", text)
    return text

def remove_related(text):
    author = "(?:[A-Z][A-Za-z'`-]+)"
    etal = "(?:et al.?)"
    additional = "(?:,? (?:(?:and |& )?" + author + "|" + etal + "))"
    year_num = "(?:19|20)[0-9][0-9]"
    page_num = "(?:, p.? [0-9]+)?"  # Always optional
    year = "(?:, *" + year_num + page_num + "| *\(" + year_num + page_num + "\))"
    # year_index = "*[a|b|c]"
    year_index = "[a|b|c]?"
    regex = "(" + author + additional + "*" + year + year_index + ")"

    matches = re.findall(regex, text)
    # print(matches)
    for m in matches:
        text = text.replace(f'({m})', '')
        text = text.replace(m, '')
    return text

text = 'variational lossy autoencoders (Chen et al., 2016b), PixelGAN autoencoders (Makhzani & Frey, 2017)'

print(remove_related(text))