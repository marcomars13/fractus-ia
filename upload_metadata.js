import { createUmi } from "@metaplex-foundation/umi-bundle-defaults";
import { irysUploader } from "@metaplex-foundation/umi-uploader-irys";
import { keypairIdentity, createSignerFromKeypair } from "@metaplex-foundation/umi";
import fs from "fs";
import path from "path";

const walletPath = "/Users/marco/my-new-keypair.json";
const imagePath = "assets/fractus_logo.png";
const metadataPath = "assets/fractus_token.json";

(async () => {
  try {
    console.log("🔹 Lecture de la clé locale...");
    const secretKey = new Uint8Array(JSON.parse(fs.readFileSync(walletPath, "utf8")));

    // ✅ Création de l'instance UMI reliée au Devnet Solana
    const umi = createUmi("https://api.devnet.solana.com").use(irysUploader());
    const keypair = umi.eddsa.createKeypairFromSecretKey(secretKey);
    const signer = createSignerFromKeypair(umi, keypair);
    umi.use(keypairIdentity(signer));

    console.log("🚀 Préparation du logo...");
    const imageBuffer = fs.readFileSync(imagePath);
    const mimeType = "image/png";
    const fileName = path.basename(imagePath);
    const imageFile = {
      buffer: imageBuffer,
      fileName,
      displayName: fileName,
      uniqueName: fileName,
      contentType: mimeType,
    };

    console.log("🚀 Upload du logo vers Arweave...");
    // ✅ Correction → l’API attend un tableau de fichiers
    const [imageUri] = await umi.uploader.upload([imageFile]);
    console.log("✅ Logo uploadé :", imageUri);

    console.log("🚀 Upload du metadata JSON...");
    const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
    metadata.image = imageUri; // insère le lien du logo
    const metadataUri = await umi.uploader.uploadJson(metadata);
    console.log("✅ Metadata uploadé :", metadataUri);

    console.log("\n🎯 URI FINAL DU TOKEN FRA :\n", metadataUri);
    console.log("➡️ Utilise cette URI pour mettre à jour ton token FRA sur Solana.");
  } catch (err) {
    console.error("❌ Erreur :", err);
  }
})();

