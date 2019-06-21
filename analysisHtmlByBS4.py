from bs4 import BeautifulSoup

def analysisHtml(html_str):
    sentences = []
    soup = BeautifulSoup(html_str, 'html.parser')
    for tag in soup.find_all('a'):
        question = tag.span.string
        if question == "" or question is None:
            continue
        else:
            sen = question.strip().replace("\n","").replace("\t","").replace(" ","").replace("\r\n","")
            if sen=="" or sen is None:
                continue
            sentences.append(sen)

    return sentences
