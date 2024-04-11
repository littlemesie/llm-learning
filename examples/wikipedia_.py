# coding=utf-8

"""Wikipedia dataset containing cleaned articles of all languages."""

import bz2
import codecs
import json
import os
import re
import xml.etree.cElementTree as etree
from urllib.parse import quote

# Source: https://en.wikipedia.org/wiki/List_of_Wikipedias (accessed 3/1/2019)
# Removed because no articles: hz.
WIKIPEDIA_LANGUAGES = ["zh"]

# Source: for each Wikipedia language code (example shown for "ab"), aliases for namespaces -2 and 6 accessed via this API call:
# https://ab.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=namespacealiases|namespaces&format=json&formatversion=2 (accessed 12/21/2021)
MEDIA_ALIASES = {}

# Source: for each Wikipedia language code (example shown for "ab"), aliases for namespace 14 accessed via this API call:
# https://ab.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=namespacealiases|namespaces&format=json&formatversion=2 (accessed 12/21/2021)
CAT_ALIASES = {}

_BASE_URL_TMPL = "https://dumps.wikimedia.org/{lang}wiki/{date}/"

def _extract_content(file):
    """Extracts article content from a single WikiMedia XML file."""
    print("Extracting content from", file)
    with open(file, "rb") as f:
        f = bz2.BZ2File(filename=f)
        # Workaround due to: https://github.com/tensorflow/tensorflow/issues/33563
        utf_f = codecs.getreader("utf-8")(f)
        context = etree.iterparse(utf_f, events=("end",))
        for unused_event, elem in context:
            if not elem.tag.endswith("page"):
                continue
            namespace = elem.tag[:-4]
            title = elem.find(f"./{namespace}title").text
            ns = elem.find(f"./{namespace}ns").text
            id_ = elem.find(f"./{namespace}id").text
            red_ = elem.find(f"./{namespace}redirect")

            # Filter pages that are not in the "main" namespace.
            if ns != "0":
                elem.clear()
                continue

            raw_content = elem.find(f"./{namespace}revision/{namespace}text").text
            elem.clear()

            # Filter redirects.
            if raw_content is None or red_ is not None:
                continue

            yield id_, title, raw_content


def _parse_and_clean_wikicode(raw_content, parser, language):
    """Strips formatting and unwanted sections from raw page content."""
    wikicode = parser.parse(raw_content)

    # Filters for magic words that are parser instructions -- e.g., __NOTOC__
    re_rm_magic = re.compile("__[A-Z]*__", flags=re.UNICODE)

    # Filters for file/image links.
    media_prefixes = "|".join(["File", "Image", "Media"] + MEDIA_ALIASES.get(language, []))
    re_rm_wikilink = re.compile(f"^(?:{media_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

    def rm_wikilink(obj):
        return bool(re_rm_wikilink.match(str(obj.title)))

    # Filters for references and tables
    def rm_tag(obj):
        return str(obj.tag) in {"ref", "table"}

    # Leave category links in-place but remove the category prefixes
    cat_prefixes = "|".join(["Category"] + CAT_ALIASES.get(language, []))
    re_clean_wikilink = re.compile(f"^(?:{cat_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

    def is_category(obj):
        return bool(re_clean_wikilink.match(str(obj.title)))

    def clean_wikilink(obj):
        text = obj.__strip__()
        text = re.sub(re_clean_wikilink, "", text)
        obj.text = text

    def try_replace_obj(obj):
        try:
            clean_wikilink(obj)
        except ValueError:
            # For unknown reasons, objects are sometimes not found.
            pass

    def try_remove_obj(obj, section):
        try:
            section.remove(obj)
        except ValueError:
            # For unknown reasons, objects are sometimes not found.
            pass

    section_text = []
    # Filter individual sections to clean.
    for section in wikicode.get_sections(flat=True, include_lead=True, include_headings=True):
        for obj in section.ifilter_wikilinks(recursive=True):
            if rm_wikilink(obj):
                try_remove_obj(obj, section)
            elif is_category(obj):
                try_replace_obj(obj)
        for obj in section.ifilter_tags(matches=rm_tag, recursive=True):
            try_remove_obj(obj, section)

        section_text.append(re.sub(re_rm_magic, "", section.strip_code().strip()))
    return "\n\n".join(section_text)


def _construct_url(title, language):
    # See: https://meta.wikimedia.org/wiki/Help:URL
    return f"https://{language}.wikipedia.org/wiki/{quote(title)}"


def _clean_content(inputs, language):
    """Cleans raw wikicode to extract text."""
    import mwparserfromhell

    id_, title, raw_content = inputs
    try:
        text = _parse_and_clean_wikicode(raw_content, parser=mwparserfromhell, language=language)
    except (mwparserfromhell.parser.ParserError) as e:
        return

    if not text:
        return

    url = _construct_url(title, language)

    return {"id": id_, "url": url, "title": title, "text": text}

def extract_content(file, path, file_name):
    """Extracts article content from a single WikiMedia XML file."""
    print("Extracting content from", file)
    data_list = []
    with open(file, "rb") as f:
        f = bz2.BZ2File(filename=f)
        # Workaround due to: https://github.com/tensorflow/tensorflow/issues/33563
        utf_f = codecs.getreader("utf-8")(f)
        context = etree.iterparse(utf_f, events=("end",))
        for unused_event, elem in context:
            if not elem.tag.endswith("page"):
                continue
            namespace = elem.tag[:-4]
            title = elem.find(f"./{namespace}title").text
            ns = elem.find(f"./{namespace}ns").text
            id_ = elem.find(f"./{namespace}id").text
            red_ = elem.find(f"./{namespace}redirect")

            # Filter pages that are not in the "main" namespace.
            if ns != "0":
                elem.clear()
                continue

            raw_content = elem.find(f"./{namespace}revision/{namespace}text").text
            elem.clear()

            # Filter redirects.
            if raw_content is None or red_ is not None:
                continue

            example = _clean_content((id_, title, raw_content), 'zh')
            data_list.append(example)
            print(example)
    # 将数据保存为JSONL格式
    with open(f'{path}/{file_name}.json', 'w') as f:
        for item in data_list:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            # break

if __name__ == '__main__':
    """"""
    path = '../data'
    file = 'zhwiki-20240320-pages-articles-multistream2.xml-p187713p630160.bz2'
    file_name = file.split('.')[0]
    extract_content(f"{path}/{file}", path, file_name)
