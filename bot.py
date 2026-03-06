"""
⚽ BOT TELEGRAM — SUPER AGENT FOOTBALL (VERSION CORRIGÉE)
- Suppression des ConversationHandler conflictuels
- Callbacks directs simples et fiables
- Gestion d'état via context.user_data
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters,
)
from telegram.constants import ParseMode
from dotenv import load_dotenv

load_dotenv()

from super_agent import FootballSuperAgent, CONFIG

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("FootballBot")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ── Super Agent singleton ──────────────────────────────────────────────
_agent: Optional[FootballSuperAgent] = None

def get_agent() -> FootballSuperAgent:
    global _agent
    if _agent is None:
        logger.info("⚽ Chargement du Super Agent...")
        _agent = FootballSuperAgent()
        logger.info("✅ Super Agent prêt !")
    return _agent

# ── Keyboards ─────────────────────────────────────────────────────────

def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚽ Pronostics",  callback_data="action:pronostics"),
         InlineKeyboardButton("🔍 Match solo",  callback_data="action:match")],
        [InlineKeyboardButton("🏆 Classement",  callback_data="action:classement"),
         InlineKeyboardButton("💰 Value Bets",  callback_data="action:value")],
        [InlineKeyboardButton("📈 Mes stats",   callback_data="action:stats")],
    ])

def kb_leagues(action: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League",   callback_data=f"{action}:PL")],
        [InlineKeyboardButton("🇪🇸 La Liga",           callback_data=f"{action}:PD")],
        [InlineKeyboardButton("🇩🇪 Bundesliga",         callback_data=f"{action}:BL1")],
        [InlineKeyboardButton("🇮🇹 Serie A",            callback_data=f"{action}:SA")],
        [InlineKeyboardButton("🇫🇷 Ligue 1",            callback_data=f"{action}:FL1")],
        [InlineKeyboardButton("🏆 Champions League",    callback_data=f"{action}:CL")],
        [InlineKeyboardButton("🔙 Retour",              callback_data="action:menu")],
    ])

# ── Helpers ────────────────────────────────────────────────────────────

def chunks(text: str, n: int = 3800):
    lines, cur = text.split("\n"), ""
    result = []
    for line in lines:
        if len(cur) + len(line) + 1 > n:
            result.append(cur)
            cur = line + "\n"
        else:
            cur += line + "\n"
    if cur:
        result.append(cur)
    return result

async def send_report(msg, text: str):
    for part in chunks(text):
        try:
            await msg.reply_text(f"```\n{part}\n```", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await msg.reply_text(part)

# ══════════════════════════════════════════════════════════════════════
#  COMMANDES
# ══════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"🏆 *Bonjour {user.first_name} !*\n\n"
        f"Je suis votre *Super Agent Pronostics Football* ⚽🤖\n\n"
        f"📊 Ratings Elo & Pi\\-ratings\n"
        f"🔮 Prédictions ML calibrées\n"
        f"💰 Détection de Value Bets\n"
        f"⚔️ Derbies & matchs clés\n"
        f"📈 Suivi de performances\n\n"
        f"👇 *Choisissez une action :*"
    )
    await update.message.reply_text(
        text, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=kb_main()
    )

async def cmd_pronostics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏟️ *Choisissez une ligue :*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_leagues("prono"),
    )

async def cmd_classement(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏆 *Classement ELO — Choisissez une ligue :*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_leagues("elo"),
    )

async def cmd_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💰 *Value Bets — Choisissez une ligue :*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_leagues("value"),
    )

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        a   = get_agent()
        acc = a.validator.generate_accountability_report()
        total   = acc.get("total_predictions", 0)
        correct = acc.get("correct", 0)
        pct     = acc.get("accuracy_pct", 0)

        if total == 0:
            text = "📈 *Mes stats*\n\n📭 Aucune prédiction archivée.\nLancez `/pronostics` pour commencer !"
        else:
            bar    = "🟢" * int(pct / 10) + "⚫" * (10 - int(pct / 10))
            trend  = "🔥 En forme !" if pct >= 55 else ("⚡ Stable" if pct >= 45 else "❄️ En dessous")
            text   = (
                f"📈 *Performances*\n\n"
                f"✅ Corrects : *{correct}/{total}*\n"
                f"🎯 Réussite : *{pct}%*\n\n"
                f"{bar}\n\n{trend}"
            )
        await update.effective_message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_main())
    except Exception as e:
        await update.effective_message.reply_text(f"❌ Erreur : {e}")

async def cmd_match(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["awaiting"] = "match_home"
    await update.message.reply_text(
        "🔍 *Match spécifique*\n\nEntrez l'équipe *domicile* :\n_ex: Arsenal, PSG, Bayern Munich_",
        parse_mode=ParseMode.MARKDOWN,
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📖 *Commandes :*\n\n"
        "/start → Menu principal\n"
        "/pronostics → Pronostics par ligue\n"
        "/match → Analyser un match précis\n"
        "/classement → Classement Elo\n"
        "/value → Value bets\n"
        "/stats → Mes performances\n"
        "/help → Cette aide"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_main())

# ══════════════════════════════════════════════════════════════════════
#  CALLBACK HANDLER UNIQUE — gère TOUS les boutons
# ══════════════════════════════════════════════════════════════════════

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data  = query.data  # ex: "prono:PL", "action:stats", "elo:FL1"

    # ── Menu principal ──────────────────────────────────────────────
    if data == "action:menu":
        await query.edit_message_text(
            "👇 *Choisissez une action :*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_main(),
        )

    elif data == "action:pronostics":
        await query.edit_message_text(
            "🏟️ *Choisissez une ligue :*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_leagues("prono"),
        )

    elif data == "action:classement":
        await query.edit_message_text(
            "🏆 *Classement ELO — Choisissez une ligue :*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_leagues("elo"),
        )

    elif data == "action:value":
        await query.edit_message_text(
            "💰 *Value Bets — Choisissez une ligue :*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_leagues("value"),
        )

    elif data == "action:stats":
        await cmd_stats(update, context)

    elif data == "action:match":
        context.user_data["awaiting"] = "match_home"
        await query.edit_message_text(
            "🔍 Entrez l'équipe *domicile* :\n_ex: Arsenal, PSG, Bayern Munich_",
            parse_mode=ParseMode.MARKDOWN,
        )

    # ── Pronostics ligue ────────────────────────────────────────────
    elif data.startswith("prono:"):
        league      = data.split(":")[1]
        league_name = CONFIG["LEAGUES"].get(league, league)
        await query.edit_message_text(
            f"⏳ *Analyse {league_name} en cours...*\n\n"
            f"🔄 Collecte des données...\n"
            f"📊 Calcul des ratings Elo & Pi...\n"
            f"🤖 Entraînement du modèle ML...\n"
            f"🔮 Génération des pronostics...\n\n"
            f"_~10 secondes_",
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            a      = get_agent()
            report = await asyncio.get_event_loop().run_in_executor(
                None, lambda: a.run(league=league, verbose=False)
            )
            await send_report(query.message, report)
            await query.message.reply_text(
                "✅ *Analyse terminée !*\n\nQue souhaitez-vous faire ensuite ?",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_main(),
            )
        except Exception as e:
            logger.error("Erreur prono : %s", e)
            await query.message.reply_text(
                f"❌ Erreur lors de l'analyse : `{e}`\nRéessayez dans quelques secondes.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb_main(),
            )

    # ── Classement ELO ──────────────────────────────────────────────
    elif data.startswith("elo:"):
        league      = data.split(":")[1]
        league_name = CONFIG["LEAGUES"].get(league, league)
        await query.edit_message_text(
            f"⏳ *Calcul classement {league_name}...*",
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            a          = get_agent()
            standings  = await asyncio.get_event_loop().run_in_executor(
                None, lambda: a.get_standings_by_elo(league)
            )
            await query.message.reply_text(
                f"```\n{standings}\n```",
                parse_mode=ParseMode.MARKDOWN,
            )
            await query.message.reply_text(
                "✅ Classement généré !",
                reply_markup=kb_main(),
            )
        except Exception as e:
            await query.message.reply_text(f"❌ Erreur : {e}", reply_markup=kb_main())

    # ── Value Bets ──────────────────────────────────────────────────
    elif data.startswith("value:"):
        league      = data.split(":")[1]
        league_name = CONFIG["LEAGUES"].get(league, league)
        await query.edit_message_text(
            f"⏳ *Recherche Value Bets {league_name}...*",
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            a = get_agent()

            def find_value_bets():
                fixtures = a.scout.get_upcoming_fixtures(league)
                history  = a.scout.get_historical_results(league)
                elo      = a.stats.compute_elo_ratings(history)
                pi       = a.stats.compute_pi_ratings(history)
                a.ml.train(history)
                found = []
                for fix in fixtures:
                    ctx  = a.news.get_match_context(fix["home_team"], fix["away_team"], league)
                    pred = a.ml.predict_match(fix["home_team"], fix["away_team"], elo, pi, history, ctx)
                    pred = a.validator.validate_probabilities(pred)
                    pred = a.validator.compute_value_bet(pred)
                    if pred.get("is_value_bet"):
                        found.append(pred)
                return found, len(fixtures)

            vbs, total = await asyncio.get_event_loop().run_in_executor(None, find_value_bets)

            if not vbs:
                text = (
                    f"💤 *Aucun value bet — {league_name}*\n\n"
                    f"Matchs analysés : {total}\n"
                    f"Seuil : edge > 5%\n\n"
                    f"_Essayez une autre ligue !_"
                )
            else:
                lines = [f"💰 *VALUE BETS — {league_name}*\n{'━'*28}\n"]
                oc_labels = {"home": "🏠 Domicile", "draw": "🤝 Match Nul", "away": "✈️ Extérieur"}
                for vb in vbs:
                    oc    = vb.get("best_value_outcome", "")
                    odds  = vb.get("bookmaker_odds", {})
                    kelly = vb.get("kelly_fraction", 0)
                    edge  = vb.get("best_value_amount", 0)
                    lines += [
                        f"⚽ *{vb['home_team']} vs {vb['away_team']}*",
                        f"  ✅ Pari : {oc_labels.get(oc, oc)}",
                        f"  📊 Edge : +{edge*100:.1f}%",
                        f"  🎯 Kelly : {kelly*100:.1f}% bankroll",
                        f"  📉 1={odds.get('home','?')}  X={odds.get('draw','?')}  2={odds.get('away','?')}",
                        "",
                    ]
                text = "\n".join(lines)

            await query.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=kb_main())

        except Exception as e:
            await query.message.reply_text(f"❌ Erreur : {e}", reply_markup=kb_main())

# ══════════════════════════════════════════════════════════════════════
#  MESSAGE HANDLER — Texte libre (pour /match)
# ══════════════════════════════════════════════════════════════════════

def detect_league(home: str, away: str) -> str:
    mapping = {
        "PL":  {"Arsenal","Manchester City","Liverpool","Chelsea","Tottenham",
                "Manchester United","Newcastle","Aston Villa","Brighton","West Ham",
                "Everton","Crystal Palace","Brentford","Fulham","Wolves"},
        "PD":  {"Real Madrid","Barcelona","Atletico Madrid","Sevilla","Valencia","Villarreal"},
        "BL1": {"Bayern Munich","Borussia Dortmund","RB Leipzig","Bayer Leverkusen","Eintracht Frankfurt"},
        "SA":  {"Inter Milan","AC Milan","Juventus","Napoli","Roma","Lazio","Atalanta"},
        "FL1": {"Paris Saint-Germain","Olympique de Marseille","Monaco",
                "Olympique Lyonnais","Lille","Nice","Rennes","Lens"},
    }
    for league, teams in mapping.items():
        if home in teams or away in teams:
            return league
    return "PL"

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text    = update.message.text.strip()
    waiting = context.user_data.get("awaiting", "")

    # ── Saisie équipe domicile ──────────────────────────────────────
    if waiting == "match_home":
        context.user_data["home_team"] = text
        context.user_data["awaiting"]  = "match_away"
        await update.message.reply_text(
            f"✅ Domicile : *{text}*\n\nMaintenant l'équipe *extérieure* :",
            parse_mode=ParseMode.MARKDOWN,
        )

    # ── Saisie équipe extérieure → lancer analyse ──────────────────
    elif waiting == "match_away":
        home = context.user_data.pop("home_team", "Team A")
        away = text
        context.user_data.pop("awaiting", None)

        await update.message.reply_text(
            f"⏳ *Analyse...*\n⚽ {home} vs {away}",
            parse_mode=ParseMode.MARKDOWN,
        )
        try:
            a      = get_agent()
            league = detect_league(home, away)

            def do_predict():
                return a.predict_single_match(home, away, league)

            result = await asyncio.get_event_loop().run_in_executor(None, do_predict)
            await update.message.reply_text(
                f"```\n{result}\n```",
                parse_mode=ParseMode.MARKDOWN,
            )
            await update.message.reply_text(
                "✅ Analyse terminée !",
                reply_markup=kb_main(),
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Erreur : {e}", reply_markup=kb_main())

    # ── Texte libre sans contexte ───────────────────────────────────
    else:
        await update.message.reply_text(
            "👋 Utilisez le menu ou tapez /start",
            reply_markup=kb_main(),
        )

# ══════════════════════════════════════════════════════════════════════
#  LANCEMENT
# ══════════════════════════════════════════════════════════════════════

async def post_init(app: Application):
    await app.bot.set_my_commands([
        BotCommand("start",      "🏠 Menu principal"),
        BotCommand("pronostics", "⚽ Pronostics de la journée"),
        BotCommand("match",      "🔍 Analyser un match précis"),
        BotCommand("classement", "🏆 Classement ELO"),
        BotCommand("value",      "💰 Value bets"),
        BotCommand("stats",      "📈 Mes performances"),
        BotCommand("help",       "📖 Aide"),
    ])
    logger.info("✅ Commandes configurées")

def main():
    if not BOT_TOKEN:
        raise ValueError("❌ TELEGRAM_BOT_TOKEN manquant dans .env !")

    logger.info("🚀 Démarrage du Bot Football...")

    # Pré-charger le Super Agent au démarrage
    get_agent()

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    # Commandes
    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("pronostics",  cmd_pronostics))
    app.add_handler(CommandHandler("classement",  cmd_classement))
    app.add_handler(CommandHandler("value",       cmd_value))
    app.add_handler(CommandHandler("stats",       cmd_stats))
    app.add_handler(CommandHandler("match",       cmd_match))
    app.add_handler(CommandHandler("help",        cmd_help))

    # UN SEUL callback handler pour tous les boutons
    app.add_handler(CallbackQueryHandler(on_callback))

    # Messages texte (pour /match)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    logger.info("✅ Bot prêt ! En attente de messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

if __name__ == "__main__":
    main()
