import { PublicKey, clusterApiUrl, Connection } from '@solana/web3.js';
import { encodeURL } from '@solana/pay';
import QRCode from 'qrcode';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import BigNumber from 'bignumber.js';  // Thêm import này

class SolanaService {
    constructor() {
        this.merchantWallet = process.env.SOLANA_MERCHANT_WALLET || 'BaGmuHhApRiTJ9Y19RyEgThzj88yDG32eHej5ywM8ko3';
        this.label = process.env.SOLANA_LABEL || 'NHOM8';
        this.network = process.env.SOLANA_NETWORK || 'devnet';
        this.currencyAPI = process.env.SOLANA_CURRENCY_API || 'https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd';
        this.usdToVndRate = parseFloat(process.env.SOLANA_USD_TO_VND_RATE) || 24500;
        this.connection = new Connection(clusterApiUrl(this.network));
    }

    async getSolanaPriceInVND() {
        try {
            const response = await axios.get(this.currencyAPI);
            const solUSD = response.data.solana.usd;
            return solUSD * this.usdToVndRate;
        } catch (error) {
            console.error('Error getting Solana price:', error);
            return 100 * this.usdToVndRate;
        }
    }

    async vndToSolana(vndAmount) {
        const solVNDPrice = await this.getSolanaPriceInVND();
        const solAmount = vndAmount / solVNDPrice;
        
        // Convert to BigNumber và round đến 6 decimal places
        return new BigNumber(solAmount).decimalPlaces(6);
    }

    async createPaymentRequest(amountVND, orderId, description = '') {
        try {
            const solAmount = await this.vndToSolana(amountVND);
            const reference = uuidv4();
            
            const recipient = new PublicKey(this.merchantWallet);
            
            // Sử dụng BigNumber cho amount
            const urlFields = {
                recipient,
                amount: solAmount,  // Đây giờ là BigNumber
                label: this.label,
                message: description || `Thanh toán đơn hàng #${orderId}`,
                memo: `Order:${orderId}`
            };

            const url = encodeURL(urlFields);
            
            return {
                paymentUrl: url.toString(),
                reference,
                solAmount: solAmount.toString(),  // Convert về string để return
                vndAmount: amountVND,
                orderId,
                recipient: this.merchantWallet,
                createdAt: new Date().toISOString()
            };
        } catch (error) {
            throw new Error(`Failed to create payment request: ${error.message}`);
        }
    }

    async generateQRCode(paymentUrl) {
        try {
            const qrCodeDataURL = await QRCode.toDataURL(paymentUrl, {
                width: 300,
                margin: 2,
                color: {
                    dark: '#000000',
                    light: '#FFFFFF'
                }
            });
            return qrCodeDataURL;
        } catch (error) {
            throw new Error(`Failed to generate QR code: ${error.message}`);
        }
    }

    async verifyPayment(signature, expectedAmount, recipientAddress) {
        try {
            const transaction = await this.connection.getTransaction(signature, {
                commitment: 'confirmed'
            });
            
            if (!transaction) {
                return { verified: false, message: 'Transaction not found' };
            }

            if (transaction.meta.err) {
                return { verified: false, message: 'Transaction failed' };
            }

            return { verified: true, transaction };
        } catch (error) {
            return { verified: false, message: error.message };
        }
    }
}

export default SolanaService;
