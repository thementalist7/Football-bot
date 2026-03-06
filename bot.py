"""
╔══════════════════════════════════════════════════════════════════════╗
║        ⚽  BOT TELEGRAM — SUPER AGENT PRONOSTICS FOOTBALL  ⚽        ║
║                                                                      ║
║  Commandes disponibles :                                             ║
║   /start        → Accueil & menu principal                          ║
║   /pronostics   → Pronostics du jour (sélection de ligue)           ║
║   /match        → Prédire un match spécifique                       ║
║   /classement   → Classement ELO d'une ligue                        ║
║   /value        → Value bets du jour                                ║
║   /stats        → Mes performances passées                          ║
║   /leagues      → Ligues disponibles                                ║
║   /help         → Aide complète                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Optional

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode
from dotenv import load_dotenv

load_dotenv()

# ── Import du Super Agent ──────────────────────────────────────────────
from super_agent import FootballSuperAgent, CONFIG

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s → %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FootballBot")

# ── Config ─────────────────────────────────────────────────────────────
BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_ID    = os.getenv("ADMIN_TELEGRAM_ID", "")      # Optionnel : votre ID perso

# ── États ConversationHandler ──────────────────────────────────────────
CHOOSE_LEAGUE, CHOOSE_MATCH_HOME, CHOOSE_MATCH_AWAY = range(3)

# ── Instance Super Agent (singleton) ──────────────────────────────────
agent: Optional[FootballSuperAgent] = None

def get_agent() -> FootballSuperAgent:
    global agent
    if agent is None:
        logger.info("⚽ Initialisation du Super Agent...")
        agent = FootballSuperAgent()
    return agent

# ══════════════════════════════════════════════════════════════════════
#  HELPERS UI
# ══════════════════════════════════════════════════════════════════════

def leagues_keyboard() -> InlineKeyboardMarkup:
    """Clavier inline de sélection de ligue."""
    buttons = [
        [InlineKeyboardButton("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League",  callback_data="league_PL")],
        [InlineKeyboardButton("🇪🇸 La Liga",          callback_data="league_PD")],
        [InlineKeyboardButton("🇩🇪 Bundesliga",        callback_data="league_BL1")],
        [InlineKeyboardButton("🇮🇹 Serie A",           callback_data="league_SA")],
        [InlineKeyboardButton("🇫🇷 Ligue 1",           callback_data="league_FL1")],
        [InlineKeyboardButton("🏆 Champions League",   callback_data="league_CL")],
    ]
    return InlineKeyboardMarkup(buttons)

def main_menu_keyboard() -> InlineKeyboardMarkup:
    """Menu principal inline."""
    buttons = [
        [
            InlineKeyboardButton("⚽ Pronostics",  callback_data="menu_pronostics"),
            InlineKeyboardButton("🔍 Match solo",  callback_data="menu_match"),
        ],
        [
            InlineKeyboardButton("📊 Classement",  callback_data="menu_classement"),
            InlineKeyboardButton("💰 Value Bets",  callback_data="menu_value"),
        ],
        [
            InlineKeyboardButton("📈 Mes stats",   callback_data="menu_stats"),
            InlineKeyboardButton("ℹ️ Aide",         callback_data="menu_help"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)

def escape_md(text: str) -> str:
    """Échappe les caractères spéciaux MarkdownV2."""
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in text)

def chunk_message(text: str, max_len: int = 4000) -> list[str]:
    """Découpe un long message en morceaux Telegram-compatibles."""
    lines  = text.split("\n")
    chunks = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line + "\n"
        else:
            current += line + "\n"
    if current:
        chunks.append(current)
    return chunks

async def send_long(update: Update, text: str, context: ContextTypes.DEFAULT_TYPE):
    """Envoie un message potentiellement long en plusieurs parties."""
    chunks = chunk_message(text)
    for i, chunk in enumerate(chunks):
        try:
            await update.effective_message.reply_text(
                f"`{chunk}`" if i == 0 else f"`{chunk}`",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            await update.effective_message.reply_text(chunk)

# ══════════════════════════════════════════════════════════════════════
#  COMMANDES PRINCIPALES
# ══════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /start — Message de bienvenue."""
    user = update.effective_user
    welcome = (
        f"🏆 *Bienvenue {user.first_name} !*\n\n"
        f"Je suis votre *Super Agent de Pronostics Football* 🤖⚽\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 *Ce que je fais :*\n"
        f"• Analyse les stats de 6 ligues majeures\n"
        f"• Calcule les ratings Elo & Pi\\-ratings\n"
        f"• Détecte les *Value Bets* 💰\n"
        f"• Calibration ML \\(CatBoost\\)\n"
        f"• Suivi de mes performances passées\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"👇 *Que voulez\\-vous faire ?*"
    )
    await update.message.reply_text(
        welcome,
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=main_menu_keyboard(),
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /help."""
    help_text = (
        "📖 *AIDE — Super Agent Football*\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 *Commandes disponibles :*\n\n"
        "/start → Menu principal\n"
        "/pronostics → Pronostics journée par ligue\n"
        "/match → Analyser un match précis\n"
        "/classement → Classement Elo d'une ligue\n"
        "/value → Value bets du moment\n"
        "/stats → Performances passées de l'agent\n"
        "/leagues → Liste des ligues disponibles\n"
        "/help → Cette aide\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ *Disclaimer :*\n"
        "Précision max documentée : ~55\\-56%\n"
        "Jouez de façon responsable\\."
    )
    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=main_menu_keyboard(),
    )

async def cmd_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /leagues — liste des ligues."""
    text = (
        "🌍 *Ligues disponibles :*\n\n"
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 `PL`  → Premier League\n"
        "🇪🇸 `PD`  → La Liga\n"
        "🇩🇪 `BL1` → Bundesliga\n"
        "🇮🇹 `SA`  → Serie A\n"
        "🇫🇷 `FL1` → Ligue 1\n"
        "🏆 `CL`  → Champions League"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /stats — performances passées."""
    await update.effective_message.reply_text("📊 *Chargement des statistiques...*", parse_mode=ParseMode.MARKDOWN)
    try:
        a = get_agent()
        acc = a.validator.generate_accountability_report()
        total  = acc.get("total_predictions", 0)
        correct= acc.get("correct", 0)
        pct    = acc.get("accuracy_pct", 0)

        if total == 0:
            text = (
                "📈 *Mes performances*\n\n"
                "📭 Aucune prédiction archivée pour le moment\\.\n\n"
                "Commencez par `/pronostics` pour générer des pronostics\\!"
            )
        else:
            bar_len = 10
            filled  = int((pct / 100) * bar_len)
            bar     = "🟢" * filled + "⚫" * (bar_len - filled)
            trend   = "🔥 En forme !" if pct >= 55 else ("⚡ Stable" if pct >= 45 else "❄️ En dessous")

            text = (
                f"📈 *Performances de l'Agent*\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ Pronostics corrects : *{correct}/{total}*\n"
                f"🎯 Taux de réussite   : *{pct}%*\n\n"
                f"Progression : {bar}\n\n"
                f"Verdict : {trend}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"_Objectif cible : 55%+_"
            )

        await update.effective_message.reply_text(
            text, parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error("Erreur /stats : %s", e)
        await update.effective_message.reply_text(f"❌ Erreur : {e}")

# ══════════════════════════════════════════════════════════════════════
#  CONVERSATION : PRONOSTICS PAR LIGUE
# ══════════════════════════════════════════════════════════════════════

async def cmd_pronostics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /pronostics — step 1 : choix de ligue."""
    msg = update.effective_message
    await msg.reply_text(
        "🏟️ *Choisissez une ligue :*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=leagues_keyboard(),
    )
    return CHOOSE_LEAGUE

async def pronostics_league_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback après sélection de ligue → lancer l'analyse."""
    query = update.callback_query
    await query.answer()

    league = query.data.replace("league_", "")
    league_name = CONFIG["LEAGUES"].get(league, league)

    await query.edit_message_text(
        f"⏳ *Analyse de la {league_name} en cours...*\n\n"
        f"🔄 Collecte des données...\n"
        f"📊 Calcul des ratings...\n"
        f"🤖 Entraînement du modèle...\n"
        f"🔮 Génération des pronostics...\n\n"
        f"_Environ 5\\-10 secondes_",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    try:
        a = get_agent()
        report = a.run(league=league, verbose=False)

        # Reformatage pour Telegram (monospace)
        chunks = chunk_message(report, max_len=3800)

        await query.message.reply_text(
            f"```\n{chunks[0]}\n```",
            parse_mode=ParseMode.MARKDOWN,
        )
        for chunk in chunks[1:]:
            await query.message.reply_text(
                f"```\n{chunk}\n```",
                parse_mode=ParseMode.MARKDOWN,
            )

        # Bouton retour menu
        await query.message.reply_text(
            "✅ *Analyse terminée !* Que souhaitez\\-vous faire ensuite ?",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=main_menu_keyboard(),
        )

    except Exception as e:
        logger.error("Erreur pronostics : %s", e)
        await query.message.reply_text(
            f"❌ *Erreur lors de l'analyse :* {escape_md(str(e))}",
            parse_mode=ParseMode.MARKDOWN_V2,
        )

    return ConversationHandler.END

# ══════════════════════════════════════════════════════════════════════
#  CONVERSATION : MATCH UNIQUE
# ══════════════════════════════════════════════════════════════════════

async def cmd_match(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /match — step 1 : demande équipe domicile."""
    await update.effective_message.reply_text(
        "🔍 *Analyse d'un match spécifique*\n\n"
        "Entrez le nom de l'*équipe domicile* :\n"
        "_(ex: Arsenal, Barcelona, Paris Saint-Germain)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    return CHOOSE_MATCH_HOME

async def match_home_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reçoit l'équipe domicile → demande équipe extérieure."""
    context.user_data["home_team"] = update.message.text.strip()
    await update.message.reply_text(
        f"✅ Équipe domicile : *{context.user_data['home_team']}*\n\n"
        f"Maintenant, entrez l'*équipe extérieure* :",
        parse_mode=ParseMode.MARKDOWN,
    )
    return CHOOSE_MATCH_AWAY

async def match_away_received(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reçoit l'équipe extérieure → lance la prédiction."""
    home = context.user_data.get("home_team", "Team A")
    away = update.message.text.strip()

    await update.message.reply_text(
        f"⏳ *Analyse en cours...*\n\n"
        f"⚽ {home} vs {away}",
        parse_mode=ParseMode.MARKDOWN,
    )

    try:
        a = get_agent()

        # Détection automatique de la ligue selon les équipes connues
        league = _detect_league(home, away)
        result = a.predict_single_match(home, away, league)

        await update.message.reply_text(
            f"```\n{result}\n```",
            parse_mode=ParseMode.MARKDOWN,
        )
        await update.message.reply_text(
            "✅ Analyse terminée ! Que souhaitez\\-vous faire ?",
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=main_menu_keyboard(),
        )

    except Exception as e:
        logger.error("Erreur match : %s", e)
        await update.message.reply_text(f"❌ Erreur : {e}")

    return ConversationHandler.END

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Annule la conversation en cours."""
    await update.effective_message.reply_text(
        "❌ Annulé. Retour au menu.",
        reply_markup=main_menu_keyboard(),
    )
    return ConversationHandler.END

def _detect_league(home: str, away: str) -> str:
    """Détecte automatiquement la ligue d'un match selon les équipes."""
    pl_teams   = {"Arsenal", "Manchester City", "Liverpool", "Chelsea", "Tottenham",
                  "Manchester United", "Newcastle", "Aston Villa", "Brighton", "West Ham",
                  "Everton", "Crystal Palace", "Brentford", "Fulham", "Wolves"}
    pd_teams   = {"Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia",
                  "Villarreal", "Real Sociedad", "Athletic Club", "Betis"}
    bl1_teams  = {"Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
                  "Eintracht Frankfurt", "Union Berlin", "Freiburg"}
    sa_teams   = {"Inter Milan", "AC Milan", "Juventus", "Napoli", "Roma",
                  "Lazio", "Atalanta", "Fiorentina", "Torino"}
    fl1_teams  = {"Paris Saint-Germain", "Olympique de Marseille", "Monaco",
                  "Olympique Lyonnais", "Lille", "Nice", "Rennes", "Lens"}

    pair = {home, away}
    for team in pair:
        if team in pl_teams:  return "PL"
        if team in pd_teams:  return "PD"
        if team in bl1_teams: return "BL1"
        if team in sa_teams:  return "SA"
        if team in fl1_teams: return "FL1"
    return "PL"  # default

# ══════════════════════════════════════════════════════════════════════
#  CONVERSATION : CLASSEMENT ELO
# ══════════════════════════════════════════════════════════════════════

async def cmd_classement(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /classement → choix de ligue."""
    await update.effective_message.reply_text(
        "🏆 *Classement ELO — Choisissez une ligue :*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=leagues_keyboard(),
    )
    return CHOOSE_LEAGUE

async def classement_league_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback classement → affiche le ranking Elo."""
    query = update.callback_query
    await query.answer()
    league = query.data.replace("league_", "")
    league_name = CONFIG["LEAGUES"].get(league, league)

    await query.edit_message_text(
        f"⏳ *Calcul du classement Elo {league_name}...*",
        parse_mode=ParseMode.MARKDOWN,
    )
    try:
        a = get_agent()
        standings = a.get_standings_by_elo(league)
        await query.message.reply_text(
            f"```\n{standings}\n```",
            parse_mode=ParseMode.MARKDOWN,
        )
        await query.message.reply_text(
            "✅ Classement généré !",
            reply_markup=main_menu_keyboard(),
        )
    except Exception as e:
        await query.message.reply_text(f"❌ Erreur : {e}")
    return ConversationHandler.END

# ══════════════════════════════════════════════════════════════════════
#  CONVERSATION : VALUE BETS
# ══════════════════════════════════════════════════════════════════════

async def cmd_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler /value — Affiche les value bets."""
    await update.effective_message.reply_text(
        "💰 *Recherche de Value Bets — Choisissez une ligue :*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=leagues_keyboard(),
    )
    return CHOOSE_LEAGUE

async def value_league_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Callback value bets → analyse et filtre les value bets."""
    query = update.callback_query
    await query.answer()
    league = query.data.replace("league_", "")
    league_name = CONFIG["LEAGUES"].get(league, league)

    await query.edit_message_text(
        f"⏳ *Recherche de Value Bets {league_name}...*",
        parse_mode=ParseMode.MARKDOWN,
    )
    try:
        a   = get_agent()
        # Lancer l'analyse complète pour extraire les value bets
        fixtures = a.scout.get_upcoming_fixtures(league)
        history  = a.scout.get_historical_results(league)
        elo      = a.stats.compute_elo_ratings(history)
        pi       = a.stats.compute_pi_ratings(history)
        a.ml.train(history)

        value_bets_found = []
        for fix in fixtures:
            ctx  = a.news.get_match_context(fix["home_team"], fix["away_team"], league)
            pred = a.ml.predict_match(fix["home_team"], fix["away_team"], elo, pi, history, ctx)
            pred = a.validator.validate_probabilities(pred)
            pred = a.validator.compute_value_bet(pred)

            if pred.get("is_value_bet"):
                value_bets_found.append(pred)

        if not value_bets_found:
            text = (
                f"💤 *Aucun value bet significatif*\n\n"
                f"Ligue : {league_name}\n"
                f"Matchs analysés : {len(fixtures)}\n\n"
                f"_Seuil : edge > {CONFIG['VALUE_BET_THRESHOLD']*100:.0f}%_"
            )
        else:
            lines = [f"💰 *VALUE BETS — {league_name}*\n", f"{'━'*30}\n"]
            for vb in value_bets_found:
                oc = vb.get("best_value_outcome", "")
                oc_labels = {
                    "home": f"🏠 Vic {vb['home_team']}",
                    "draw": "🤝 Match Nul",
                    "away": f"✈️ Vic {vb['away_team']}",
                }
                odds = vb.get("bookmaker_odds", {})
                kelly = vb.get("kelly_fraction", 0)
                edge  = vb.get("best_value_amount", 0)
                lines += [
                    f"⚽ *{vb['home_team']} vs {vb['away_team']}*",
                    f"  ✅ Bet : {oc_labels.get(oc, oc)}",
                    f"  📊 Edge : +{edge*100:.1f}%",
                    f"  🎯 Kelly : {kelly*100:.1f}% bankroll",
                    f"  📉 Cotes : 1={odds.get('home','?')} X={odds.get('draw','?')} 2={odds.get('away','?')}",
                    "",
                ]
            text = "\n".join(lines)

        await query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        await query.message.reply_text(
            "✅ Analyse terminée !",
            reply_markup=main_menu_keyboard(),
        )
    except Exception as e:
        logger.error("Erreur value bets : %s", e)
        await query.message.reply_text(f"❌ Erreur : {e}")
    return ConversationHandler.END

# ══════════════════════════════════════════════════════════════════════
#  CALLBACKS DU MENU PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gère les clics sur le menu principal."""
    query = update.callback_query
    await query.answer()
    action = query.data.replace("menu_", "")

    if action == "pronostics":
        await query.edit_message_text(
            "🏟️ *Choisissez une ligue :*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=leagues_keyboard(),
        )
        context.user_data["callback_action"] = "pronostics"

    elif action == "classement":
        await query.edit_message_text(
            "🏆 *Classement ELO — Choisissez une ligue :*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=leagues_keyboard(),
        )
        context.user_data["callback_action"] = "classement"

    elif action == "value":
        await query.edit_message_text(
            "💰 *Value Bets — Choisissez une ligue :*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=leagues_keyboard(),
        )
        context.user_data["callback_action"] = "value"

    elif action == "match":
        await query.edit_message_text(
            "🔍 *Match spécifique*\n\nUtilisez la commande `/match` pour analyser un match précis.",
            parse_mode=ParseMode.MARKDOWN,
        )

    elif action == "stats":
        await cmd_stats(update, context)

    elif action == "help":
        help_text = (
            "📖 *Commandes disponibles :*\n\n"
            "/start → Menu principal\n"
            "/pronostics → Pronostics par ligue\n"
            "/match → Analyser un match précis\n"
            "/classement → Classement Elo\n"
            "/value → Value bets\n"
            "/stats → Mes performances\n"
            "/leagues → Ligues disponibles\n"
            "/help → Cette aide"
        )
        await query.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def league_callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Route les callbacks de sélection de ligue selon l'action en cours."""
    query = update.callback_query
    action = context.user_data.get("callback_action", "pronostics")

    if action == "classement":
        await classement_league_chosen(update, context)
    elif action == "value":
        await value_league_chosen(update, context)
    else:
        await pronostics_league_chosen(update, context)

# ══════════════════════════════════════════════════════════════════════
#  HANDLER TEXTE LIBRE — Analyse express
# ══════════════════════════════════════════════════════════════════════

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Répond aux messages texte libres avec le menu."""
    text = update.message.text.lower().strip()

    # Détection de patterns courants
    if any(w in text for w in ["prono", "pronostic", "prédiction", "match"]):
        await update.message.reply_text(
            "⚽ Choisissez une ligue pour les pronostics :",
            reply_markup=leagues_keyboard(),
        )
        context.user_data["callback_action"] = "pronostics"
    elif any(w in text for w in ["value", "bet", "cote", "pari"]):
        await update.message.reply_text(
            "💰 Choisissez une ligue pour les value bets :",
            reply_markup=leagues_keyboard(),
        )
        context.user_data["callback_action"] = "value"
    elif any(w in text for w in ["elo", "classement", "ranking"]):
        await update.message.reply_text(
            "🏆 Choisissez une ligue :",
            reply_markup=leagues_keyboard(),
        )
        context.user_data["callback_action"] = "classement"
    else:
        await update.message.reply_text(
            f"👋 Bonjour ! Voici ce que je peux faire :",
            reply_markup=main_menu_keyboard(),
        )

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION & LANCEMENT DU BOT
# ══════════════════════════════════════════════════════════════════════

async def post_init(application: Application):
    """Configure les commandes du bot après initialisation."""
    commands = [
        BotCommand("start",       "🏠 Menu principal"),
        BotCommand("pronostics",  "⚽ Pronostics de la journée"),
        BotCommand("match",       "🔍 Analyser un match précis"),
        BotCommand("classement",  "🏆 Classement ELO"),
        BotCommand("value",       "💰 Value bets"),
        BotCommand("stats",       "📈 Mes performances"),
        BotCommand("leagues",     "🌍 Ligues disponibles"),
        BotCommand("help",        "📖 Aide"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("✅ Commandes du bot configurées")

def main():
    """Point d'entrée principal du bot."""
    if not BOT_TOKEN:
        raise ValueError(
            "❌ TELEGRAM_BOT_TOKEN manquant !\n"
            "   1. Créez un bot via @BotFather sur Telegram\n"
            "   2. Copiez le token dans votre fichier .env\n"
            "      TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
        )

    logger.info("🚀 Démarrage du Bot Football Telegram...")
    logger.info("📡 Token détecté : %s...%s", BOT_TOKEN[:8], BOT_TOKEN[-4:])

    # Pré-charger le Super Agent
    get_agent()

    # Construction de l'application
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # ── ConversationHandler : Pronostics ──
    pronostics_conv = ConversationHandler(
        entry_points=[CommandHandler("pronostics", cmd_pronostics)],
        states={
            CHOOSE_LEAGUE: [
                CallbackQueryHandler(pronostics_league_chosen, pattern=r"^league_")
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_message=False,
    )

    # ── ConversationHandler : Match unique ──
    match_conv = ConversationHandler(
        entry_points=[CommandHandler("match", cmd_match)],
        states={
            CHOOSE_MATCH_HOME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, match_home_received)
            ],
            CHOOSE_MATCH_AWAY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, match_away_received)
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_message=False,
    )

    # ── ConversationHandler : Classement ──
    classement_conv = ConversationHandler(
        entry_points=[CommandHandler("classement", cmd_classement)],
        states={
            CHOOSE_LEAGUE: [
                CallbackQueryHandler(classement_league_chosen, pattern=r"^league_")
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_message=False,
    )

    # ── ConversationHandler : Value Bets ──
    value_conv = ConversationHandler(
        entry_points=[CommandHandler("value", cmd_value)],
        states={
            CHOOSE_LEAGUE: [
                CallbackQueryHandler(value_league_chosen, pattern=r"^league_")
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        per_message=False,
    )

    # ── Enregistrement des handlers ──
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("leagues", cmd_leagues))
    app.add_handler(CommandHandler("stats",   cmd_stats))
    app.add_handler(pronostics_conv)
    app.add_handler(match_conv)
    app.add_handler(classement_conv)
    app.add_handler(value_conv)

    # Menu principal callbacks
    app.add_handler(CallbackQueryHandler(menu_callback,          pattern=r"^menu_"))
    app.add_handler(CallbackQueryHandler(league_callback_router, pattern=r"^league_"))

    # Texte libre
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # ── Lancement ──
    logger.info("✅ Bot prêt ! Polling en cours...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )

if __name__ == "__main__":
    main()
