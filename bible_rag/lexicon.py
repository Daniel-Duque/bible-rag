"""Topic lexicon: modern words -> the words each translation actually uses.

Questions use today's vocabulary ("alcohol"), the texts don't (KJV "wine", "strong drink";
RV1909 "vino", "sidra"). A topic lists those words per language, grouped into *concepts*, so
we can (1) collect every passage on the topic, (2) see when two translations word the same
passage differently (KJV "hell" / RV1909 "sepulcro" / WEB "Sheol"), and (3) search for the two
opposite claims a "contradiction" question is about.

Term syntax (matched on lowercase, accent-free text, see `fold`):
  word      whole word              strong drink   phrase
  word*     prefix (drunk*)         word?          homograph: counts only when confirmed by a
                                                   parallel passage in another language
                                                   (Spanish "vino" is also "came")
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_MARKS = re.compile("[\u0300-\u036f]")


def fold(text: str) -> str:
    """Lowercase, accents removed (RV1909 writes "á Dios"). Same length as the input, so a
    match in the folded text can be cut out of the original."""
    out = _MARKS.sub("", unicodedata.normalize("NFD", text)).lower()
    if len(out) != len(text):  # rare characters: the slow way keeps offsets aligned
        out = "".join((unicodedata.normalize("NFD", c)[0].lower() or c)[:1] for c in text)
    return out


def term_pattern(term: str) -> str:
    body = re.escape(term.rstrip("*?")).replace(r"\ ", r"\s+")
    return rf"\b{body}\w*" if term.rstrip("?").endswith("*") else rf"\b{body}\b"


def compile_terms(terms) -> re.Pattern | None:
    terms = list(terms)
    return re.compile("|".join(term_pattern(t) for t in terms)) if terms else None


@dataclass(frozen=True)
class Concept:
    """One idea and the words that express it in each language."""

    gloss: str  # shown in notes, e.g. "the grave"
    en: tuple[str, ...] = ()
    es: tuple[str, ...] = ()

    def terms(self, language: str) -> tuple[str, ...]:
        return getattr(self, language, ())


@dataclass(frozen=True)
class Note:
    """A verified remark about how a translation words this topic."""

    en: str
    es: str
    versions: tuple[str, ...] = ()  # shown only when one of these version ids is loaded


@dataclass(frozen=True)
class Topic:
    key: str
    name: dict  # {"en": ..., "es": ...} as shown to the user
    examples: tuple[str, ...]  # what the topic is about, in plain words: matched by meaning
    concepts: tuple[Concept, ...]
    # the two opposite statements searched for: {"en": (A, B), "es": (A, B)}.
    # None: generic "X is good" / "X is evil" (see GENERIC_CLAIMS)
    claims: dict | None = None
    sides: dict | None = None  # side titles {"en": (A, B), "es": (A, B)}
    notes: tuple[Note, ...] = ()
    # words that must also appear in a passage for it to be on the topic ("unclean" is about
    # food only next to "eat", "meat"...): {"en": "eat*, meat*", "es": ...}
    context: dict = field(default_factory=dict)

    def context_terms(self, language: str) -> list[str]:
        return [t for t in self.context.get(language, "").split(", ") if t]

    def terms(self, language: str, ambiguous: bool = True) -> list[str]:
        return [
            t for c in self.concepts for t in c.terms(language) if ambiguous or not t.endswith("?")
        ]


def _c(gloss: str, en: str, es: str) -> Concept:
    return Concept(gloss, tuple(en.split(", ")) if en else (), tuple(es.split(", ")) if es else ())


def _pair(en_a, en_b, es_a, es_b) -> dict:
    return {"en": (en_a, en_b), "es": (es_a, es_b)}


# Every term was checked against the KJV, RV1909 and WEB texts. RV1909 spelling is archaic
# but accents are ignored when matching.
TOPICS = [
    Topic(
        "alcohol",
        {"en": "alcohol (wine, strong drink)", "es": "el alcohol (vino, sidra)"},
        (
            "alcohol",
            "drinking wine",
            "drunkenness",
            "beer and liquor",
            "strong drink",
            "el alcohol",
            "beber vino",
            "la borrachera",
            "las bebidas alcohólicas",
        ),
        (
            _c("wine", "wine, wines, new wine, winebibber*", "vino?, vinos, mosto"),
            # KJV "Beer", "Beer-sheba" are places: "beer" (WEB) needs a confirming parallel
            _c("strong drink", "strong drink, beer?", "sidra, cerveza"),
            _c("drunkenness", "drunk?, drunken*, drunkard*", "embriag*, borrach*"),
        ),
        claims=_pair(
            "wine is good: a blessing and a gift of God that makes the heart of man glad",
            "wine is a mocker: woe to the drunkard; be not drunk with wine, it leads to sin",
            "el vino es bueno: bendición y don de Dios que alegra el corazón del hombre",
            "el vino es escarnecedor: ay del borracho; no os embriaguéis, lleva al pecado",
        ),
        sides=_pair(
            "Passages that speak favourably of wine",
            "Passages that warn against it",
            "Pasajes favorables al vino",
            "Pasajes que advierten en contra",
        ),
        notes=(
            Note(
                "In the RV1909 “alcohol” is kohl, the eye paint (2 Reyes 9:30, Ezequiel 23:40), "
                "not a drink: where it says Jezebel “adornó sus ojos con alcohol”, the KJV says "
                "she “painted her face”.",
                "En la RV1909 «alcohol» es el kohl, la pintura de ojos (2 Reyes 9:30, Ezequiel "
                "23:40), no una bebida: donde dice que Jezabel «adornó sus ojos con alcohol», la "
                "KJV dice «painted her face».",
                ("rv1909",),
            ),
            Note(
                "Spanish “vino” is also the verb “came” (venir): a passage that only matches "
                "“vino” is kept when its parallel in another translation mentions the topic.",
                "«Vino» también es el verbo venir: un pasaje que sólo coincide por «vino» se "
                "conserva si su paralelo en otra traducción trata el tema.",
                ("rv1909",),
            ),
        ),
    ),
    Topic(
        "money",
        {"en": "money and riches", "es": "el dinero y las riquezas"},
        (
            "money",
            "wealth and riches",
            "being rich",
            "greed",
            "poverty and wealth",
            "el dinero",
            "las riquezas",
            "ser rico",
            "la avaricia",
        ),
        (
            _c("money", "money, mammon", "dinero*, mammon"),
            _c("riches", "rich, riches, wealth*, treasure*", "rico, ricos, riqueza*, tesoro*"),
            _c("greed", "covet*, greed*", "codici*, avaricia*, avaro*"),
        ),
        claims=_pair(
            "riches are a blessing of the LORD: he gives power to get wealth, and wisdom and "
            "riches",
            "woe unto you that are rich: the love of money is the root of all evil; it is hard "
            "for a rich man to enter the kingdom",
            "las riquezas son bendición de Jehová: él da el poder para hacer las riquezas",
            "ay de vosotros ricos: el amor del dinero es la raíz de todos los males; difícil "
            "es que un rico entre en el reino",
        ),
        sides=_pair(
            "Passages where wealth is a blessing",
            "Passages that warn against wealth",
            "Pasajes donde la riqueza es bendición",
            "Pasajes que advierten contra la riqueza",
        ),
    ),
    Topic(
        "divorce",
        {"en": "divorce", "es": "el divorcio"},
        (
            "divorce",
            "putting away a wife",
            "separation of husband and wife",
            "el divorcio",
            "repudiar a la esposa",
            "separarse del cónyuge",
        ),
        (_c("divorce", "divorce*, put away?, putteth away?, putting away?", "divorcio*, repudi*"),),
        claims=_pair(
            "divorce is allowed: a man may write a bill of divorcement and put away his wife",
            "divorce is forbidden: what God has joined let no man put asunder; whoever puts away "
            "his wife commits adultery",
            "el divorcio es permitido: el hombre puede dar carta de divorcio y repudiar a su mujer",
            "el divorcio es prohibido: lo que Dios juntó no lo aparte el hombre; el que repudia a "
            "su mujer comete adulterio",
        ),
        sides=_pair(
            "Passages that allow it",
            "Passages that forbid it",
            "Pasajes que lo permiten",
            "Pasajes que lo prohíben",
        ),
    ),
    Topic(
        "war",
        {"en": "war and violence", "es": "la guerra y la violencia"},
        (
            "war",
            "violence",
            "killing",
            "fighting enemies",
            "peace and pacifism",
            "la guerra",
            "la violencia",
            "matar",
            "la paz",
        ),
        (
            _c("war", "war, wars, battle*", "guerra*, batalla*"),
            _c(
                "killing",
                "kill*, slay*, slew, slain, murder*",
                "matar, matare, mataras, mataron, matad, matado, homicid*, asesin*",
            ),
            _c("peace", "peace, peaceable, peacemakers", "paz, pacific*"),
        ),
        claims=_pair(
            "the LORD said: go up and fight against them, for I have delivered them into thine "
            "hand; smite them and utterly destroy them",
            "thou shalt not kill; love your enemies; they shall beat their swords into "
            "plowshares, nation shall not lift up sword against nation",
            "Jehová dijo: subid y pelead contra ellos, porque yo los he entregado en tu mano; "
            "heridlos y destruidlos del todo",
            "no matarás; amad a vuestros enemigos; volverán sus espadas en azadones, no alzará "
            "espada gente contra gente",
        ),
        sides=_pair(
            "Passages that command or approve war",
            "Passages that call for peace",
            "Pasajes que mandan o aprueban la guerra",
            "Pasajes que llaman a la paz",
        ),
    ),
    Topic(
        "hell",
        {"en": "hell and the grave", "es": "el infierno y el sepulcro"},
        (
            "hell",
            "the afterlife",
            "eternal punishment",
            "the grave",
            "Sheol and Hades",
            "el infierno",
            "el castigo eterno",
            "el sepulcro",
            "la vida después de la muerte",
        ),
        (
            _c("hell", "hell", "infierno*"),
            _c("the grave", "grave, graves, sepulchre*, tomb*", "sepulcro*, sepultura*, tumba*"),
            _c("Sheol", "sheol, shiul", "seol"),
            _c("Hades", "hades", "hades"),
            _c("Gehenna", "gehenna, gihano", "gehenna"),
            _c("perdition", "perdition, damnation", "perdicion, condenacion"),
        ),
        claims=_pair(
            "the wicked are cast into hell, everlasting fire and punishment",
            "the dead know nothing; they sleep in the grave and perish",
            "los impíos son echados al infierno, al fuego eterno y al castigo",
            "los muertos nada saben; duermen en el sepulcro y perecen",
        ),
        sides=_pair(
            "Passages about everlasting punishment",
            "Passages about death as sleep or an end",
            "Pasajes sobre castigo eterno",
            "Pasajes sobre la muerte como sueño o fin",
        ),
        notes=(
            Note(
                "The KJV writes “hell” for the Hebrew Sheol and for the Greek Hades and Gehenna "
                "(Psalm 16:10, Matthew 5:22); the RV1909 says “sepulcro” or “infierno”, and the "
                "WEB keeps “Sheol”, “Hades” and “Gehenna”.",
                "La KJV dice «hell» tanto para el hebreo Seol como para el griego Hades y Gehenna "
                "(Salmos 16:10, Mateo 5:22); la RV1909 dice «sepulcro» o «infierno», y la WEB "
                "conserva «Sheol», «Hades» y «Gehenna».",
            ),
        ),
    ),
    Topic(
        "love",
        {"en": "love and hate", "es": "el amor y el odio"},
        (
            "love",
            "hate",
            "does God hate people",
            "loving your enemies",
            "charity",
            "el amor",
            "el odio",
            "¿Dios aborrece a alguien?",
            "amar a los enemigos",
        ),
        (
            _c(
                "love",
                "love, loved, loveth, lovest, loving*, lovingkindness",
                "amor, amores, amar, amaras, amara, amad, amado, amados, amada, amaba, amas, aman, "
                "amaste",
            ),
            _c("charity", "charity", "caridad"),
            _c("hate", "hate, hated, hateth, hatest, hatred", "aborrec*, odio*"),
        ),
        claims=_pair(
            "God is love; love your enemies and your neighbour",
            "God hates the workers of iniquity; I hated Esau",
            "Dios es amor; amad a vuestros enemigos y a vuestro prójimo",
            "Dios aborrece a los que obran iniquidad; a Esaú aborrecí",
        ),
        sides=_pair(
            "Passages about love",
            "Passages about hatred",
            "Pasajes sobre el amor",
            "Pasajes sobre el aborrecimiento",
        ),
        notes=(
            Note(
                "The KJV's “charity” (1 Corinthians 13) renders the Greek word it elsewhere "
                "translates “love” (agape); the RV1909 says “caridad” and the WEB “love”.",
                "La «charity» de la KJV (1 Corintios 13) traduce la palabra griega que en otros "
                "lugares vierte «love» (agape); la RV1909 dice «caridad» y la WEB «love».",
            ),
        ),
    ),
    Topic(
        "faith_works",
        {"en": "faith and works", "es": "la fe y las obras"},
        (
            "faith and works",
            "salvation by faith",
            "justification",
            "grace and good works",
            "la fe y las obras",
            "la salvación por fe",
            "la justificación",
            "la gracia",
        ),
        (
            _c("faith", "faith", "fe"),
            _c("works", "works, deeds", "obras"),
            _c("justified", "justif*", "justific*"),
            _c("grace", "grace", "gracia"),
        ),
        claims=_pair(
            "a man is justified by faith without the works of the law; saved by grace through "
            "faith, not of works",
            "by works a man is justified, and not by faith only; faith without works is dead",
            "el hombre es justificado por la fe sin las obras de la ley; salvos por gracia por la "
            "fe, no por obras",
            "el hombre es justificado por las obras, y no solamente por la fe; la fe sin obras es "
            "muerta",
        ),
        sides=_pair(
            "Passages: by faith, not works",
            "Passages: works are necessary",
            "Pasajes: por la fe, no por obras",
            "Pasajes: las obras son necesarias",
        ),
    ),
    Topic(
        "seeing_god",
        {"en": "seeing God", "es": "ver a Dios"},
        (
            "seeing God",
            "has anyone seen God",
            "seeing God face to face",
            "ver a Dios",
            "¿alguien ha visto a Dios?",
            "ver a Dios cara a cara",
        ),
        (
            _c(
                "seeing God",
                "seen god, see god, saw god, saw the god, seen the lord, saw the "
                "lord, seen the face of god",
                "visto a dios, ver a dios, vi a dios, vio a dios, vieron a dios, vieron al dios, "
                "vio jamas a dios, visto a jehova, vi a jehova",
            ),
            _c("face to face", "face to face", "cara a cara"),
        ),
        claims=_pair(
            "I have seen God face to face, and my life is preserved; they saw God, and did eat "
            "and drink",
            "no man hath seen God at any time; no man shall see my face and live",
            "vi a Dios cara a cara, y fue librada mi alma; vieron a Dios, y comieron y bebieron",
            "a Dios nadie le vio jamás; no me verá hombre y vivirá",
        ),
        sides=_pair(
            "Passages where people see God",
            "Passages saying no one can see God",
            "Pasajes donde alguien ve a Dios",
            "Pasajes que dicen que nadie puede ver a Dios",
        ),
    ),
    Topic(
        "temptation",
        {"en": "God tempting or testing", "es": "Dios que tienta o prueba"},
        (
            "temptation",
            "does God tempt people",
            "God testing people",
            "la tentación",
            "¿Dios tienta a las personas?",
            "Dios prueba a los hombres",
        ),
        (
            _c("temptation", "tempt*", "tent*, tienta*"),
            _c("testing", "prove, proved, proveth, tested, test, trieth, tried", "prob*, prueba*"),
        ),
        claims=_pair(
            "God did tempt Abraham; the LORD your God proveth you",
            "let no man say he is tempted of God: God cannot be tempted, neither tempteth he any "
            "man",
            "tentó Dios a Abraham; Jehová vuestro Dios os prueba",
            "nadie diga que es tentado de Dios: Dios no puede ser tentado, ni él tienta a alguno",
        ),
        sides=_pair(
            "Passages where God tempts or tests",
            "Passages saying God tempts no one",
            "Pasajes donde Dios tienta o prueba",
            "Pasajes que dicen que Dios no tienta a nadie",
        ),
        notes=(
            Note(
                "Genesis 22:1: the KJV says God did “tempt” Abraham and the RV1909 “tentó”, while "
                "the WEB says “tested”; James 1:13 says God tempts no one. The Hebrew verb "
                "(nasah) means to test or prove.",
                "Génesis 22:1: la KJV dice que Dios «tempted» a Abraham y la RV1909 «tentó», "
                "mientras la WEB dice «tested»; Santiago 1:13 dice que Dios no tienta a nadie. El "
                "verbo hebreo (nasah) significa probar.",
            ),
        ),
    ),
    Topic(
        "repentance",
        {"en": "God changing his mind", "es": "Dios que se arrepiente"},
        (
            "does God change his mind",
            "God repenting",
            "God regrets",
            "¿Dios cambia de opinión?",
            "Dios se arrepiente",
        ),
        (
            _c("repent", "repent*", "arrepent*, arrepint*"),
            _c("be sorry", "sorry", "peso?"),
        ),
        claims=_pair(
            "the LORD repented of the evil which he thought to do; it repented the LORD that he "
            "had made man",
            "God is not a man, that he should repent; I am the LORD, I change not",
            "Jehová se arrepintió del mal que dijo que había de hacer; arrepintióse Jehová de "
            "haber hecho hombre",
            "Dios no es hombre para que se arrepienta; yo Jehová, no me mudo",
        ),
        sides=_pair(
            "Passages where God repents",
            "Passages saying God does not change",
            "Pasajes donde Dios se arrepiente",
            "Pasajes que dicen que Dios no cambia",
        ),
        notes=(
            Note(
                "Genesis 6:6: “it repented the LORD” (KJV) and “arrepintióse Jehová” (RV1909) are "
                "rendered “Yahweh was sorry” in the WEB.",
                "Génesis 6:6: «it repented the LORD» (KJV) y «arrepintióse Jehová» (RV1909) se "
                "vierten «Yahweh was sorry» en la WEB.",
            ),
        ),
    ),
    Topic(
        "oaths",
        {"en": "oaths and swearing", "es": "los juramentos"},
        (
            "oaths",
            "swearing",
            "making vows",
            "los juramentos",
            "jurar",
            "hacer votos",
        ),
        (_c("oath", "swear*, sware, sworn, oath*", "jur*"), _c("vow", "vow*", "voto*")),
        claims=_pair(
            "thou shalt swear by his name; he that sweareth by the LORD shall glory",
            "swear not at all; let your yea be yea and your nay be nay",
            "por su nombre jurarás; el que jura por Jehová será alabado",
            "no juréis en ninguna manera; sea vuestro hablar sí, sí; no, no",
        ),
        sides=_pair(
            "Passages that command or allow oaths",
            "Passages that forbid swearing",
            "Pasajes que mandan o permiten jurar",
            "Pasajes que prohíben jurar",
        ),
    ),
    Topic(
        "anger",
        {"en": "anger", "es": "la ira"},
        (
            "anger",
            "wrath",
            "being angry",
            "la ira",
            "el enojo",
            "enojarse",
        ),
        (_c("anger", "anger, angry, wrath*", "ira, enoj*, airad*, sana"),),
        claims=_pair(
            "be ye angry and sin not; the anger of the LORD was kindled",
            "let all bitterness, wrath and anger be put away; the wrath of man worketh not the "
            "righteousness of God",
            "airaos y no pequéis; el furor de Jehová se encendió",
            "toda amargura, enojo e ira sea quitada de vosotros; la ira del hombre no obra la "
            "justicia de Dios",
        ),
        sides=_pair(
            "Passages where anger is allowed or divine",
            "Passages that reject anger",
            "Pasajes donde la ira se permite o es divina",
            "Pasajes que rechazan la ira",
        ),
    ),
    Topic(
        "slavery",
        {"en": "slavery", "es": "la esclavitud"},
        (
            "slavery",
            "slaves and masters",
            "servants",
            "la esclavitud",
            "esclavos y amos",
            "los siervos",
        ),
        (
            _c(
                "bondage",
                "bondman, bondmen, bondmaid*, bondwoman, bondservant*, bondage, slave*",
                "esclav*, servidumbre",
            ),
            _c("servants", "servant*, maidservant*", "siervo*, sierva*"),
        ),
        claims=_pair(
            "servants, be obedient to your masters; ye may buy bondmen, they shall be your "
            "possession for ever",
            "proclaim liberty throughout the land; there is neither bond nor free; masters, give "
            "unto your servants that which is just",
            "siervos, obedeced a vuestros amos; compraréis siervos, y los poseeréis para siempre",
            "pregonad libertad en la tierra; no hay siervo ni libre; amos, haced lo que es justo "
            "con vuestros siervos",
        ),
        sides=_pair(
            "Passages that regulate or accept slavery",
            "Passages about liberty and equality",
            "Pasajes que regulan o aceptan la esclavitud",
            "Pasajes sobre libertad e igualdad",
        ),
        notes=(
            Note(
                "The KJV and the RV1909 usually write “servant” / “siervo” for the Hebrew “ebed” "
                "and the Greek “doulos”, words that also mean “slave”.",
                "La KJV y la RV1909 suelen escribir «servant» / «siervo» para el hebreo «ebed» y "
                "el griego «doulos», palabras que también significan «esclavo».",
            ),
        ),
    ),
    Topic(
        "food",
        {"en": "clean and unclean food", "es": "los alimentos limpios e inmundos"},
        (
            "eating pork",
            "clean and unclean food",
            "dietary laws",
            "eating meat",
            "comer cerdo",
            "alimentos limpios e inmundos",
            "las leyes de comida",
        ),
        (
            _c("unclean", "unclean", "inmundo, inmunda, inmundos, inmundas"),
            _c("swine", "swine, pig, pigs", "puerco*, cerdo*"),
            _c("clean", "clean", "limpio, limpia, limpios, limpias"),
            _c("food", "meat, meats, food, foods", "vianda*, comida*, manjar*"),
        ),
        claims=_pair(
            "every creature of God is good, and nothing to be refused, if it be received with "
            "thanksgiving; all meats are clean",
            "the swine is unclean to you: of their flesh shall ye not eat; these are unclean "
            "unto you",
            "todo lo que Dios crió es bueno, y nada hay que desechar, tomándose con hacimiento "
            "de gracias; todas las viandas son limpias",
            "el puerco os será inmundo: de la carne de ellos no comeréis; éstos os serán inmundos",
        ),
        sides=_pair(
            "Passages that allow all foods",
            "Passages that forbid unclean foods",
            "Pasajes que permiten todo alimento",
            "Pasajes que prohíben alimentos inmundos",
        ),
        context={
            "en": "eat, eateth, eaten, eating, ate, meat, meats, food, foods, flesh, creature*, "
            "beast*",
            "es": "comer, comera*, comere*, comeis, comed, comio, comido, comia*, comiere*, "
            "vianda*, carne, carnes, comida*, manjar*, animal*, bestia*",
        },
        notes=(
            Note(
                "In the KJV “meat” means food in general: Genesis 1:29 says the plants are given "
                "“for meat”, where the WEB says “food” and the RV1909 “para comer”.",
                "En la KJV «meat» significa alimento en general: Génesis 1:29 dice que las "
                "plantas se dan «for meat», donde la WEB dice «food» y la RV1909 «para comer».",
            ),
        ),
    ),
    Topic(
        "sabbath",
        {"en": "the sabbath", "es": "el sábado"},
        (
            "the sabbath",
            "keeping the day of rest",
            "working on Saturday",
            "el sábado",
            "guardar el día de reposo",
        ),
        (_c("sabbath", "sabbath*", "sabado*"), _c("rest", "rest", "reposo")),
        claims=_pair(
            "remember the sabbath day to keep it holy; whosoever doeth any work therein shall be "
            "put to death",
            "the sabbath was made for man, not man for the sabbath; let no man judge you in "
            "respect of the sabbath days",
            "acuérdate del día del reposo para santificarlo; cualquiera que hiciere obra en él "
            "morirá",
            "el sábado por causa del hombre es hecho, no el hombre por causa del sábado; nadie "
            "os juzgue en días de fiesta o sábados",
        ),
        sides=_pair(
            "Passages that enforce the sabbath",
            "Passages that relativise it",
            "Pasajes que imponen el sábado",
            "Pasajes que lo relativizan",
        ),
    ),
    Topic(
        "lying",
        {"en": "lying and deception", "es": "la mentira y el engaño"},
        (
            "lying",
            "deception",
            "telling the truth",
            "la mentira",
            "mentir",
            "el engaño",
        ),
        (
            _c(
                "lying",
                "lie?, lies, lying?, lied, liar*",
                "mentir*, mentira*, menti*, mintio, miente*",
            ),
            _c("deceit", "deceit*, deceiv*", "engan*"),
        ),
        claims=_pair(
            "the midwives deceived Pharaoh and God dealt well with them; Rahab hid the spies and "
            "was saved",
            "lying lips are abomination to the LORD; lie not one to another",
            "las parteras engañaron a Faraón y Dios les hizo bien; Rahab escondió a los espías y "
            "fue salvada",
            "los labios mentirosos son abominación a Jehová; no mintáis los unos a los otros",
        ),
        sides=_pair(
            "Passages where deception is spared or rewarded",
            "Passages that forbid lying",
            "Pasajes donde el engaño es perdonado o premiado",
            "Pasajes que prohíben mentir",
        ),
    ),
]

GENERIC_CLAIMS = _pair(
    "{t} is good and blessed by God",
    "{t} is evil and forbidden by God",
    "{t} es bueno y bendecido por Dios",
    "{t} es malo y prohibido por Dios",
)
GENERIC_SIDES = _pair(
    "Passages that speak favourably",
    "Passages that warn against it",
    "Pasajes favorables",
    "Pasajes que advierten en contra",
)


def _stem(word: str) -> str:
    """dancing, danced, dances -> danc; bailes -> bail (then matched as a prefix)."""
    for suffix in ("ing", "ed", "es", "s", "a", "o", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def adhoc_topic(words: list[str], language: str, expansions: dict | None = None) -> Topic:
    """A topic for a subject the lexicon doesn't know: the question's words (and their other
    forms) plus, per language, the words the translations use for it, found by meaning
    (`expansions`, see agent.Planner._expand)."""
    terms = {lang: list(ts) for lang, ts in (expansions or {}).items()}
    own = [f"{_stem(w)}*" if len(w) >= 5 else w for w in words]
    terms[language] = list(dict.fromkeys(own + terms.get(language, [])))
    name = " ".join(words)
    concept = Concept(name, tuple(terms.get("en", ())), tuple(terms.get("es", ())))
    return Topic("adhoc", {"en": name, "es": name}, (name,), (concept,))
