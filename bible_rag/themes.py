"""Biblical themes that bear on subjects the Bible never names.

The Bible says nothing about cellphones, video games or cryptocurrency, but it does speak of
things they have to do with: letters and messengers, being mastered by a habit, the use of
time, quick riches. A question about such a subject (see semantic.Semantics.absence) is
answered with the themes closest to it in meaning, each shown with *how* it relates, so that
the answer never claims the Bible speaks of the subject itself.

Each theme has:
- example phrases, modern and ancient, matched by meaning against the question's subject
  (so "pager" or "Telegram" find "communication" without being listed);
- a `link`: in what respect a subject relates to it ("in so far as it is a way of sending
  words and news to others"), shown with the passages;
- a statement in the Bible's register, searched by meaning in every translation;
- anchor passages: verses that are known to speak to the theme, checked against the KJV and
  RV1909 texts (tests/test_themes.py). They are found first; the search fills the rest.

A `frame` theme is not about a subject but about deciding on things the Bible doesn't name
("all things are lawful, but not all things edify"): it is added when the question asks
whether something is right or allowed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    key: str
    name: dict  # {"en": ..., "es": ...}
    scope: dict  # what the Bible has on it, shown next to the name
    link: dict  # how a subject relates to it (completes "How this relates to X: ...")
    examples: tuple[str, ...]  # matched by meaning
    query: dict  # searched by meaning in each translation
    anchors: tuple[str, ...] = ()  # references ("2 John 1:12"), found first
    frame: bool = False  # principles for deciding, added to questions of right and wrong

    def title(self, language: str) -> str:
        name = self.name[language]
        return f"{name[:1].upper()}{name[1:]}: {self.scope[language]}"


def _t(en: str, es: str) -> dict:
    return {"en": en, "es": es}


# Every anchor was read in the KJV and RV1909: it must speak to the theme in its own context
# (popular proof-texts that don't, e.g. Daniel 12:4 for technology, are left out).
THEMES = [
    Theme(
        "communication",
        _t("communication", "la comunicación"),
        _t("letters, messengers and news from afar", "cartas, mensajeros y noticias de lejos"),
        _t(
            "in so far as it is a way of sending words and news to others",
            "en cuanto es un medio de enviar palabras y noticias a otros",
        ),
        (
            "communication and messages",
            "phone calls, texting and email",
            "sending letters and messengers",
            "news from far away",
            "talking to someone who is far away",
            "la comunicación y los mensajes",
            "llamadas, mensajes de texto y correo",
            "cartas y mensajeros",
        ),
        _t(
            "I had many things to write unto you, but I trust to come unto you and speak face "
            "to face; good news from a far country",
            "tenía muchas cosas que escribiros, mas espero ir a vosotros y hablar boca a boca; "
            "buenas nuevas de lejanas tierras",
        ),
        ("2 John 1:12", "3 John 1:13-14", "Proverbs 25:25", "2 Thessalonians 2:2"),
    ),
    Theme(
        "speech",
        _t("the tongue and our words", "la lengua y las palabras"),
        _t("what we say and write about others", "lo que decimos y escribimos de otros"),
        _t(
            "in so far as words are spoken or written through it",
            "en cuanto por él se dicen o escriben palabras",
        ),
        (
            "the tongue and our words",
            "gossip and slander",
            "comments, posts and insults",
            "idle and careless words",
            "what we say about other people",
            "la lengua y las palabras",
            "los chismes y la calumnia",
            "comentarios e insultos",
        ),
        _t(
            "let no corrupt communication proceed out of your mouth; death and life are in the "
            "power of the tongue",
            "ninguna palabra torpe salga de vuestra boca; la muerte y la vida están en poder de "
            "la lengua",
        ),
        ("Ephesians 4:29", "James 3:5-6", "Proverbs 18:21", "Matthew 12:36", "Proverbs 10:19"),
    ),
    Theme(
        "self_control",
        _t("self-control", "el dominio propio"),
        _t("not being mastered by anything", "no dejarse dominar por nada"),
        _t(
            "in so far as it can become a habit that masters a person",
            "en cuanto puede volverse un hábito que domina a la persona",
        ),
        (
            "self-control and addiction",
            "being hooked on something",
            "compulsive habits and dependence",
            "phone and screen addiction",
            "being a slave to a habit",
            "el dominio propio y las adicciones",
            "estar enganchado a algo",
            "la dependencia y los vicios",
        ),
        _t(
            "all things are lawful unto me, but I will not be brought under the power of any; "
            "of whom a man is overcome, of the same is he brought in bondage",
            "todas las cosas me son lícitas, mas yo no me meteré debajo de potestad de nada; "
            "el que es de alguno vencido, es sujeto a la servidumbre",
        ),
        ("1 Corinthians 6:12", "2 Peter 2:19", "Proverbs 25:28", "1 Corinthians 9:27"),
    ),
    Theme(
        "time",
        _t("the use of time", "el uso del tiempo"),
        _t("redeeming the time, numbering our days", "redimir el tiempo, contar nuestros días"),
        _t("in so far as it takes up our time", "en cuanto ocupa nuestro tiempo"),
        (
            "the use of time",
            "wasting hours",
            "screen time and distraction",
            "being busy and distracted",
            "spending the day on entertainment",
            "el uso del tiempo",
            "perder el tiempo",
            "las distracciones",
        ),
        _t(
            "walk circumspectly, redeeming the time; teach us to number our days",
            "andad avisadamente, redimiendo el tiempo; enséñanos a contar nuestros días",
        ),
        ("Ephesians 5:15-16", "Psalms 90:12", "Colossians 4:5", "Ecclesiastes 3:1"),
    ),
    Theme(
        "eyes",
        _t("what we set before our eyes", "lo que ponemos delante de los ojos"),
        _t("the eyes, images and what we dwell on", "los ojos, las imágenes y lo que pensamos"),
        _t(
            "in so far as it puts images and stories before our eyes",
            "en cuanto pone imágenes e historias delante de nuestros ojos",
        ),
        (
            "what we watch and look at",
            "watching shows and films",
            "images on a screen",
            "the lust of the eyes",
            "what fills our mind",
            "lo que miramos",
            "ver series y películas",
            "las imágenes en una pantalla",
        ),
        _t(
            "I will set no wicked thing before mine eyes; whatsoever things are pure, think on "
            "these things",
            "no pondré delante de mis ojos cosa injusta; todo lo puro, en esto pensad",
        ),
        ("Psalms 101:3", "Matthew 6:22-23", "Philippians 4:8", "1 John 2:16"),
    ),
    Theme(
        "body",
        _t("the body as a temple", "el cuerpo como templo"),
        _t("caring for the body God gave", "cuidar el cuerpo que Dios dio"),
        _t("in so far as it affects the body", "en cuanto afecta al cuerpo"),
        (
            "the body and health",
            "harming one's own body",
            "taking care of one's health",
            "what we put into the body: food, smoke, drugs",
            "damaging one's health with a habit",
            "exercise and fitness",
            "el cuerpo y la salud",
            "dañar el propio cuerpo",
            "cuidar la salud",
        ),
        _t(
            "your body is the temple of the Holy Ghost, ye are not your own: glorify God in "
            "your body",
            "vuestro cuerpo es templo del Espíritu Santo, no sois vuestros: glorificad a Dios "
            "en vuestro cuerpo",
        ),
        ("1 Corinthians 6:19-20", "Romans 12:1", "1 Corinthians 3:16-17", "1 Timothy 4:8"),
    ),
    Theme(
        "idols",
        _t("idolatry", "la idolatría"),
        _t("what takes the place of God", "lo que ocupa el lugar de Dios"),
        _t(
            "in so far as it can take the first place in the heart",
            "en cuanto puede ocupar el primer lugar en el corazón",
        ),
        (
            "idols and idolatry",
            "what takes God's place",
            "devotion to the work of men's hands",
            "being obsessed with something",
            "serving two masters",
            "la idolatría",
            "lo que ocupa el lugar de Dios",
            "la obra de manos de hombres",
        ),
        _t(
            "thou shalt have no other gods before me; their idols are the work of men's hands; "
            "no man can serve two masters",
            "no tendrás dioses ajenos delante de mí; sus ídolos son obra de manos de hombres; "
            "ninguno puede servir a dos señores",
        ),
        ("Exodus 20:3-4", "Psalms 115:4-8", "Matthew 6:24", "1 John 5:21"),
    ),
    Theme(
        "skill",
        _t("tools and human skill", "las herramientas y la habilidad humana"),
        _t("craftsmen, inventions and what people build", "artífices, inventos y obras humanas"),
        _t(
            "in so far as it is a human invention or tool",
            "en cuanto es una invención o herramienta humana",
        ),
        (
            "tools and inventions",
            "technology and machines",
            "craftsmanship and skill",
            "what human beings can build",
            "engineering and devices",
            "las herramientas y los inventos",
            "la tecnología y las máquinas",
            "la habilidad de los artífices",
        ),
        _t(
            "I have filled him with wisdom and understanding in all manner of workmanship; now "
            "nothing will be restrained from them which they have imagined to do",
            "lo he henchido de sabiduría e inteligencia en todo artificio; nada les retraerá "
            "de lo que han pensado hacer",
        ),
        ("Exodus 31:3-5", "Genesis 4:22", "Genesis 11:6", "Ecclesiastes 7:29"),
    ),
    Theme(
        "wisdom",
        _t("wisdom and knowledge", "la sabiduría y el conocimiento"),
        _t("getting understanding, and its limits", "adquirir inteligencia, y sus límites"),
        _t(
            "in so far as it is about knowledge and information",
            "en cuanto trata de conocimiento e información",
        ),
        (
            "wisdom and knowledge",
            "learning and information",
            "science and education",
            "books and study",
            "knowing many things",
            "la sabiduría y el conocimiento",
            "el estudio y la información",
            "la ciencia y la educación",
        ),
        _t(
            "wisdom is the principal thing: get understanding; if any of you lack wisdom, let "
            "him ask of God; knowledge puffeth up",
            "sabiduría ante todo: adquiere inteligencia; si alguno tiene falta de sabiduría, "
            "pídala a Dios; la ciencia hincha",
        ),
        ("Proverbs 4:7", "James 1:5", "1 Corinthians 8:1", "Ecclesiastes 12:12"),
    ),
    Theme(
        "stewardship",
        _t("stewardship of money", "la administración del dinero"),
        _t("using well what is entrusted to us", "usar bien lo que se nos confía"),
        _t(
            "in so far as it is a way of using money and possessions",
            "en cuanto es una forma de usar el dinero y los bienes",
        ),
        (
            "managing money",
            "investing and saving",
            "planning and counting the cost",
            "the parable of the talents",
            "financial decisions",
            "administrar el dinero",
            "invertir y ahorrar",
            "los talentos",
        ),
        _t(
            "he delivered unto his servants his goods, to every man according to his ability; "
            "which of you sitteth not down first and counteth the cost",
            "entregó a sus siervos sus bienes, a cada uno conforme a su facultad; ¿quién no "
            "se sienta primero y echa la cuenta de los gastos?",
        ),
        ("Matthew 25:14-15", "Luke 14:28", "Proverbs 21:5", "Luke 16:10-11", "Ecclesiastes 11:2"),
    ),
    Theme(
        "debt",
        _t("debt and lending", "las deudas y los préstamos"),
        _t("borrowing, lending and surety", "pedir prestado, prestar y salir fiador"),
        _t(
            "in so far as it involves borrowing and owing",
            "en cuanto implica pedir prestado y deber",
        ),
        (
            "debt and loans",
            "borrowing money",
            "credit and interest",
            "owing money to others",
            "guaranteeing someone's debt",
            "las deudas y los préstamos",
            "pedir dinero prestado",
            "el crédito y los intereses",
        ),
        _t(
            "the borrower is servant to the lender; owe no man any thing, but to love one another",
            "el que toma prestado, siervo es del que empresta; no debáis a nadie nada, sino "
            "amaros unos a otros",
        ),
        ("Proverbs 22:7", "Romans 13:8", "Psalms 37:21", "Proverbs 22:26-27"),
    ),
    Theme(
        "chance",
        _t("chance and quick riches", "el azar y las riquezas rápidas"),
        _t("lots, and haste to be rich", "las suertes y el afán de enriquecerse"),
        _t(
            "in so far as it relies on chance or promises quick gain",
            "en cuanto depende del azar o promete ganancias rápidas",
        ),
        (
            "gambling and betting",
            "the lottery",
            "getting rich quickly",
            "risky speculation",
            "casting lots",
            "los juegos de azar y las apuestas",
            "la lotería",
            "hacerse rico rápido",
        ),
        _t(
            "wealth gotten by vanity shall be diminished; he that maketh haste to be rich shall "
            "not be innocent; the lot is cast into the lap",
            "las riquezas de vanidad disminuirán; el que se apresura a enriquecer no será sin "
            "culpa; la suerte se echa en el seno",
        ),
        ("Proverbs 13:11", "Proverbs 28:20", "1 Timothy 6:9-10", "Proverbs 16:33"),
    ),
    Theme(
        "play",
        _t("play, joy and leisure", "el juego, la alegría y el ocio"),
        _t("laughter, feasting and rest", "la risa, las fiestas y el descanso"),
        _t(
            "in so far as it is play, leisure or entertainment",
            "en cuanto es juego, ocio o entretenimiento",
        ),
        (
            "games and play",
            "entertainment and fun",
            "sports and hobbies",
            "leisure and free time",
            "laughter and celebration",
            "los juegos y la diversión",
            "el entretenimiento",
            "los deportes y el ocio",
        ),
        _t(
            "a time to laugh and a time to dance; the streets shall be full of boys and girls "
            "playing; a man hath no better thing than to be merry",
            "tiempo de reír y tiempo de bailar; las calles estarán llenas de muchachos y "
            "muchachas que jugarán; no hay bien para el hombre sino que se alegre",
        ),
        ("Ecclesiastes 3:4", "Zechariah 8:5", "Ecclesiastes 8:15", "Mark 6:31"),
    ),
    Theme(
        "purity",
        _t("sexual purity", "la pureza sexual"),
        _t(
            "lust, fornication and faithfulness", "la concupiscencia, la fornicación y la fidelidad"
        ),
        _t("in so far as it touches sexual morality", "en cuanto toca la moral sexual"),
        (
            "sexual morality",
            "lust and pornography",
            "adultery and fornication",
            "sexual temptation",
            "la pureza sexual",
            "la pornografía y la lujuria",
            "el adulterio y la fornicación",
        ),
        _t(
            "this is the will of God, your sanctification: abstain from fornication; whosoever "
            "looketh on a woman to lust after her hath committed adultery in his heart",
            "la voluntad de Dios es vuestra santificación: que os apartéis de fornicación; "
            "cualquiera que mira a una mujer para codiciarla, ya adulteró en su corazón",
        ),
        ("1 Thessalonians 4:3-5", "Matthew 5:28", "Hebrews 13:4", "1 Corinthians 6:18"),
    ),
    Theme(
        "marriage",
        _t("finding a spouse", "buscar cónyuge"),
        _t("courtship and marriage", "el noviazgo y el matrimonio"),
        _t(
            "in so far as it is a way of meeting a husband or wife",
            "en cuanto es un medio de conocer esposo o esposa",
        ),
        (
            "finding a husband or wife",
            "dating and courtship",
            "choosing a spouse",
            "arranged marriages and matchmaking",
            "meeting someone to marry",
            "buscar esposo o esposa",
            "el noviazgo",
            "elegir pareja",
        ),
        _t(
            "it is not good that the man should be alone; whoso findeth a wife findeth a good "
            "thing; be ye not unequally yoked",
            "no es bueno que el hombre esté solo; el que halló esposa halló el bien; no os "
            "juntéis en yugo con los infieles",
        ),
        ("Genesis 24:12-14", "Genesis 2:18", "Proverbs 18:22", "2 Corinthians 6:14"),
    ),
    Theme(
        "life",
        _t("life in the womb and children", "la vida en el vientre y los hijos"),
        _t("conception, barrenness and birth", "la concepción, la esterilidad y el nacimiento"),
        _t(
            "in so far as it concerns conception, pregnancy or children",
            "en cuanto concierne a la concepción, el embarazo o los hijos",
        ),
        (
            "pregnancy and conception",
            "infertility and having children",
            "life in the womb",
            "family planning",
            "unborn babies",
            "el embarazo y la concepción",
            "la infertilidad",
            "la vida en el vientre",
        ),
        _t(
            "thou hast covered me in my mother's womb; before I formed thee in the belly I "
            "knew thee; children are an heritage of the LORD",
            "tú me cubriste en el vientre de mi madre; antes que te formase en el vientre te "
            "conocí; herencia de Jehová son los hijos",
        ),
        ("Psalms 139:13-16", "Jeremiah 1:5", "Psalms 127:3", "Genesis 30:1-2"),
    ),
    Theme(
        "healing",
        _t("healing and physicians", "la sanidad y los médicos"),
        _t("remedies, physicians and trust in God", "remedios, médicos y confianza en Dios"),
        _t(
            "in so far as it is a way of treating sickness",
            "en cuanto es un medio de tratar la enfermedad",
        ),
        (
            "medicine and doctors",
            "treating illness",
            "remedies and cures",
            "surgery and hospitals",
            "mental health treatment",
            "la medicina y los médicos",
            "tratar una enfermedad",
            "los remedios",
        ),
        _t(
            "they that are whole need not a physician, but they that are sick; take a lump of "
            "figs and lay it on the boil",
            "los que están sanos no necesitan médico, sino los enfermos; tomad masa de higos "
            "y ponedla sobre la llaga",
        ),
        ("Luke 5:31", "2 Kings 20:7", "1 Timothy 5:23", "Jeremiah 8:22", "2 Chronicles 16:12"),
    ),
    Theme(
        "creation_care",
        _t("caring for the earth", "el cuidado de la tierra"),
        _t("the land, the trees and the creatures", "la tierra, los árboles y los animales"),
        _t(
            "in so far as it affects the earth and its creatures",
            "en cuanto afecta a la tierra y sus criaturas",
        ),
        (
            "the environment and nature",
            "pollution and the climate",
            "taking care of the planet",
            "animals and the land",
            "el medio ambiente",
            "la contaminación y el clima",
            "cuidar el planeta",
        ),
        _t(
            "the LORD God put the man into the garden to dress it and to keep it; the earth is "
            "the LORD's, and the fulness thereof",
            "Jehová Dios puso al hombre en el huerto para que lo labrara y lo guardase; de "
            "Jehová es la tierra y su plenitud",
        ),
        ("Genesis 2:15", "Psalms 24:1", "Leviticus 25:23-24", "Deuteronomy 20:19"),
    ),
    Theme(
        "heavens",
        _t("the heavens and the stars", "los cielos y las estrellas"),
        _t("the work of God's fingers", "la obra de los dedos de Dios"),
        _t(
            "in so far as it concerns the sky and the universe",
            "en cuanto concierne al cielo y al universo",
        ),
        (
            "the stars and the universe",
            "outer space",
            "astronomy and the planets",
            "the moon and the sun",
            "las estrellas y el universo",
            "el espacio exterior",
            "la astronomía",
        ),
        _t(
            "the heavens declare the glory of God; when I consider thy heavens, the moon and "
            "the stars, what is man",
            "los cielos cuentan la gloria de Dios; cuando veo tus cielos, la luna y las "
            "estrellas, ¿qué es el hombre?",
        ),
        ("Psalms 19:1", "Psalms 8:3-4", "Isaiah 40:26", "Job 38:31-33"),
    ),
    Theme(
        "creatures",
        _t("great creatures", "las grandes criaturas"),
        _t("behemoth, leviathan and the beasts", "behemot, leviatán y las bestias"),
        _t(
            "in so far as it concerns great or strange creatures",
            "en cuanto concierne a criaturas grandes o extrañas",
        ),
        (
            "huge ancient animals",
            "monsters and giant beasts",
            "behemoth and leviathan",
            "sea monsters and dragons",
            "extinct creatures",
            "animales enormes y antiguos",
            "monstruos y bestias gigantes",
        ),
        _t(
            "behold now behemoth, which I made with thee; canst thou draw out leviathan with "
            "an hook; God created great whales",
            "he aquí ahora behemot, al cual yo hice contigo; ¿sacarás tú al leviatán con el "
            "anzuelo?; crió Dios las grandes ballenas",
        ),
        ("Job 40:15-19", "Job 41:1", "Genesis 1:21", "Psalms 104:25-26"),
    ),
    Theme(
        "nations",
        _t("the nations", "las naciones"),
        _t(
            "every people to the ends of the earth",
            "todos los pueblos hasta lo último de la tierra",
        ),
        _t(
            "in so far as it is a nation or people of the earth",
            "en cuanto es una nación o un pueblo de la tierra",
        ),
        (
            "countries and nations",
            "a country far away",
            "the peoples of the world",
            "a nation that did not exist then",
            "continents and distant lands",
            "los países y las naciones",
            "un país lejano",
            "los pueblos del mundo",
        ),
        _t(
            "he hath made of one blood all nations of men, and determined the bounds of their "
            "habitation; go ye and teach all nations, unto the uttermost part of the earth",
            "de una sangre ha hecho todo el linaje de los hombres y les ha prefijado los términos "
            "de su habitación; id y doctrinad a todos los gentiles, hasta lo último de la tierra",
        ),
        ("Acts 17:26", "Matthew 28:19", "Revelation 7:9", "Psalms 22:27", "Acts 1:8"),
    ),
    Theme(
        "weapons",
        _t("weapons and defence", "las armas y la defensa"),
        _t("the sword, protection and peace", "la espada, la protección y la paz"),
        _t(
            "in so far as it is a weapon or a means of defence",
            "en cuanto es un arma o un medio de defensa",
        ),
        (
            "weapons",
            "self-defence",
            "owning arms",
            "protecting one's family",
            "swords and spears",
            "las armas",
            "la defensa propia",
            "proteger a la familia",
        ),
        _t(
            "he that hath no sword, let him sell his garment and buy one; all they that take "
            "the sword shall perish with the sword",
            "el que no tiene espada, venda su capa y compre una; todos los que tomaren espada, "
            "a espada perecerán",
        ),
        ("Luke 22:36-38", "Matthew 26:52", "Exodus 22:2", "Nehemiah 4:17-18", "Isaiah 2:4"),
    ),
    Theme(
        "government",
        _t("rulers and authority", "los gobernantes y la autoridad"),
        _t("choosing leaders and obeying them", "elegir a los que gobiernan y obedecerles"),
        _t(
            "in so far as it concerns rulers, laws and public life",
            "en cuanto concierne a gobernantes, leyes y vida pública",
        ),
        (
            "government and politics",
            "voting and elections",
            "choosing leaders",
            "laws and the state",
            "citizens and authorities",
            "el gobierno y la política",
            "votar y las elecciones",
            "las leyes y el estado",
        ),
        _t(
            "provide out of all the people able men, such as fear God, and place such over "
            "them; let every soul be subject unto the higher powers",
            "provee de entre todo el pueblo varones de virtud, temerosos de Dios, y ponlos "
            "sobre ellos; toda alma se someta a las potestades superiores",
        ),
        ("Exodus 18:21", "Deuteronomy 1:13", "Romans 13:1", "Proverbs 29:2", "Acts 5:29"),
    ),
    Theme(
        "work",
        _t("work and diligence", "el trabajo y la diligencia"),
        _t("labour done as to the Lord", "el trabajo hecho como para el Señor"),
        _t(
            "in so far as it concerns work and earning a living",
            "en cuanto concierne al trabajo y al sustento",
        ),
        (
            "jobs and careers",
            "working hard",
            "laziness",
            "earning a living",
            "machines doing people's work",
            "el trabajo y el empleo",
            "la pereza",
            "ganarse la vida",
        ),
        _t(
            "whatsoever ye do, do it heartily, as to the Lord; if any would not work, neither "
            "should he eat",
            "todo lo que hagáis, hacedlo de ánimo, como al Señor; si alguno no quisiere "
            "trabajar, tampoco coma",
        ),
        ("Colossians 3:23", "2 Thessalonians 3:10", "Proverbs 6:6", "Ecclesiastes 9:10"),
    ),
    Theme(
        "appearance",
        _t("outward appearance", "la apariencia exterior"),
        _t("adornment, beauty and the heart", "el adorno, la belleza y el corazón"),
        _t(
            "in so far as it concerns how we look",
            "en cuanto concierne a nuestro aspecto",
        ),
        (
            "beauty and looks",
            "fashion and makeup",
            "cosmetic treatments",
            "body image",
            "how one looks to others",
            "la belleza y el aspecto",
            "la moda y el maquillaje",
            "la imagen corporal",
        ),
        _t(
            "man looketh on the outward appearance, but the LORD looketh on the heart; whose "
            "adorning let it not be that outward adorning",
            "el hombre mira lo que está delante de sus ojos, mas Jehová mira el corazón; el "
            "adorno no sea exterior",
        ),
        ("1 Samuel 16:7", "1 Peter 3:3-4", "1 Timothy 2:9-10", "Proverbs 31:30"),
    ),
    Theme(
        "falsehood",
        _t("false reports", "los falsos rumores"),
        _t(
            "testing what we hear before passing it on", "examinar lo que se oye antes de repetirlo"
        ),
        _t(
            "in so far as reports that may be false spread through it",
            "en cuanto por él se difunden noticias que pueden ser falsas",
        ),
        (
            "fake news and rumours",
            "scams and deception",
            "believing everything we read",
            "spreading false reports",
            "las noticias falsas",
            "los rumores y las estafas",
            "creer todo lo que se lee",
        ),
        _t(
            "thou shalt not raise a false report; the simple believeth every word, but the "
            "prudent man looketh well to his going",
            "no admitirás falso rumor; el simple cree a toda palabra, mas el avisado entiende "
            "sus pasos",
        ),
        ("Exodus 23:1", "Proverbs 14:15", "1 John 4:1", "Proverbs 18:17"),
    ),
    Theme(
        "substances",
        _t("drunkenness and sobriety", "la embriaguez y la sobriedad"),
        _t("what clouds the mind", "lo que nubla el entendimiento"),
        _t(
            "in so far as it alters the mind as strong drink does",
            "en cuanto altera la mente como la bebida fuerte",
        ),
        (
            "drugs and intoxication",
            "getting high",
            "being under the influence",
            "sobriety",
            "las drogas",
            "drogarse",
            "la sobriedad",
        ),
        _t(
            "be not drunk with wine, wherein is excess; be sober, be vigilant",
            "no os embriaguéis de vino, en lo cual hay disolución; sed templados y velad",
        ),
        ("Ephesians 5:18", "Proverbs 23:29-32", "1 Peter 5:8", "Galatians 5:19-21"),
    ),
    Theme(
        "fellowship",
        _t("fellowship", "la comunión"),
        _t("gathering together and friendship", "congregarse y la amistad"),
        _t(
            "in so far as it shapes how people meet and keep company",
            "en cuanto influye en cómo las personas se reúnen y se acompañan",
        ),
        (
            "community and friendship",
            "loneliness and isolation",
            "meeting together",
            "online and in-person church",
            "la comunidad y la amistad",
            "la soledad",
            "reunirse",
        ),
        _t(
            "not forsaking the assembling of ourselves together; two are better than one",
            "no dejando nuestra congregación; mejores son dos que uno",
        ),
        ("Hebrews 10:24-25", "Acts 2:42", "Ecclesiastes 4:9-10", "Proverbs 27:17"),
    ),
    Theme(
        "pride",
        _t("pride and seeking praise", "el orgullo y buscar alabanza"),
        _t("being seen of men, and boasting", "ser vistos de los hombres, y jactarse"),
        _t(
            "in so far as it can feed the wish to be seen and praised",
            "en cuanto puede alimentar el deseo de ser visto y alabado",
        ),
        (
            "showing off and vanity",
            "seeking likes and praise",
            "comparing ourselves with others",
            "boasting",
            "la vanidad y la vanagloria",
            "buscar aprobación",
            "compararse con otros",
        ),
        _t(
            "let another man praise thee, and not thine own mouth; do not your alms before men, "
            "to be seen of them",
            "alábete el extraño, y no tu boca; no hagáis vuestra justicia delante de los "
            "hombres, para ser vistos de ellos",
        ),
        ("Proverbs 27:2", "Matthew 6:1", "Galatians 6:4", "Philippians 2:3"),
    ),
    Theme(
        "travel",
        _t("journeys and haste", "los viajes y la prisa"),
        _t(
            "going from city to city, if the Lord will",
            "ir de ciudad en ciudad, si el Señor quiere",
        ),
        _t("in so far as it is a way of travelling", "en cuanto es un medio de viajar"),
        (
            "travel and journeys",
            "vehicles and transport",
            "driving fast",
            "ships and chariots",
            "los viajes",
            "los vehículos y el transporte",
            "conducir",
        ),
        _t(
            "ye that say, to day or to morrow we will go into such a city: ye ought to say, if "
            "the Lord will; the LORD shall preserve thy going out and thy coming in",
            "los que decís: hoy y mañana iremos a tal ciudad; debierais decir: si el Señor "
            "quisiere; Jehová guardará tu salida y tu entrada",
        ),
        ("James 4:13-15", "Psalms 121:8", "Proverbs 19:2", "Psalms 107:23-24"),
    ),
    Theme(
        "rest",
        _t("rest and holy days", "el descanso y los días santos"),
        _t("the sabbath and ceasing from work", "el día de reposo y cesar del trabajo"),
        _t(
            "in so far as it concerns rest or the Lord's day",
            "en cuanto concierne al descanso o al día del Señor",
        ),
        (
            "rest and the day of rest",
            "Sunday and the sabbath",
            "taking a break from work",
            "el descanso",
            "el domingo y el día de reposo",
        ),
        _t(
            "the sabbath was made for man, and not man for the sabbath; come ye apart and rest "
            "a while",
            "el sábado por causa del hombre es hecho; venid aparte y reposad un poco",
        ),
        ("Exodus 20:8-10", "Mark 2:27", "Mark 6:31", "Romans 14:5-6"),
    ),
    Theme(
        "conscience",
        _t(
            "deciding on what the Bible doesn't name",
            "decidir sobre lo que la Biblia no nombra",
        ),
        _t("freedom, conscience and the good of others", "libertad, conciencia y el bien de otros"),
        _t(
            "the Bible's own principles for choices it doesn't name",
            "los principios de la Biblia para decisiones que no nombra",
        ),
        (
            "Is it a sin to use this?",
            "Should a Christian do this?",
            "Is it allowed for a believer?",
            "Is it wrong to have one?",
            "Is it OK for Christians?",
            "Can I use it?",
            "¿Es pecado usarlo?",
            "¿Puedo usarlo?",
            "¿Debe un cristiano hacerlo?",
            "¿Está mal tenerlo?",
        ),
        _t(
            "all things are lawful for me, but all things are not expedient; whatsoever ye do, "
            "do all to the glory of God; whatsoever is not of faith is sin",
            "todo me es lícito, mas no todo conviene; hacedlo todo a gloria de Dios; todo lo "
            "que no es de fe, es pecado",
        ),
        (
            "1 Corinthians 10:23",
            "1 Corinthians 10:31",
            "Romans 14:23",
            "1 Thessalonians 5:21",
            "1 Corinthians 8:9",
        ),
        frame=True,
    ),
]

THEME_BY_KEY = {t.key: t for t in THEMES}


def theme_examples(theme: Theme) -> list[str]:
    """What a theme is about, in words to compare subjects (or, for a frame, questions)
    with."""
    if theme.frame:
        return list(theme.examples)
    return list(dict.fromkeys([*theme.examples, theme.name["en"], theme.name["es"]]))
