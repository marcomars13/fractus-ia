from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# ======================================================
# 🔐 Bot Fractys Licence (FTSLicenceBot)
# Gère l'enregistrement des adresses Solana
# et les commandes /start + /access
# ======================================================

BOT_TOKEN = "8221302436:AAEV-l5AdYqdGmtfSv-_T4ex8lPppUcaNgE"

# === Commande /start ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Bienvenue sur le bot FTS Licence.\n"
        "Tapez /access pour activer votre licence Fractys."
    )

# === Commande /access ===
async def access(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔐 Envoyez votre **adresse Solana** pour obtenir votre accès à Fractys Access.\n\n"
        "⚠️ Une seule adresse par utilisateur sera validée."
    )

# === Gestion des messages texte (adresse Solana) ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if len(text) in [43, 44] and all(c.isalnum() for c in text):
        await update.message.reply_text(
            f"✅ Adresse enregistrée : `{text}`\n🧠 Validation en cours...",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            "⚠️ Format d’adresse invalide, veuillez vérifier votre adresse Solana."
        )

# === Lancement du bot ===
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("access", access))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot FTS Licence actif (commande /access prête)...")
    app.run_polling()

if __name__ == "__main__":
    main()

