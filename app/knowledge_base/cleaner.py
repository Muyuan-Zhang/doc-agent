import re
import unicodedata

from app.knowledge_base.parser import Section

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


class TextCleaner:
    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = _CONTROL_RE.sub("", text)
        lines = [_MULTI_SPACE_RE.sub(" ", line).rstrip() for line in text.splitlines()]
        text = "\n".join(lines)
        text = _MULTI_BLANK_RE.sub("\n\n", text)
        return text.strip()

    def clean_section(self, section: Section) -> Section:
        return Section(
            section_id=section.section_id,
            heading=self.clean(section.heading) if section.heading else None,
            content=self.clean(section.content),
        )
