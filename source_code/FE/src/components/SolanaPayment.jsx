import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';

const SolanaPayment = ({ paymentData, orderId, backendUrl, token, onPaymentSuccess, onPaymentCancel }) => {
    const [paymentStatus, setPaymentStatus] = useState('pending');
    const [loading, setLoading] = useState(false);

    const handlePaymentComplete = async () => {
        const signature = prompt('Nhập Transaction Signature từ ví Solana của bạn:');
        
        if (!signature) {
            toast.error('Vui lòng nhập Transaction Signature');
            return;
        }

        setLoading(true);
        try {
            const response = await axios.post(
                backendUrl + '/api/order/verifySolana',
                { orderId, signature },
                { headers: { token } }
            );

            if (response.data.success) {
                setPaymentStatus('success');
                toast.success('Thanh toán thành công!');
                setTimeout(() => {
                    onPaymentSuccess();
                }, 2000);
            } else {
                toast.error(response.data.message);
            }
        } catch (error) {
            console.error('Payment verification error:', error);
            toast.error('Lỗi xác minh thanh toán');
        } finally {
            setLoading(false);
        }
    };

    if (paymentStatus === 'success') {
        return (
            <div className="text-center p-8">
                <div className="text-6xl mb-4">✅</div>
                <h3 className="text-xl font-bold mb-4">Thanh toán thành công!</h3>
                <p>Đơn hàng #{orderId} đã được thanh toán</p>
                <p className="text-gray-600">Đang chuyển hướng...</p>
            </div>
        );
    }

    return (
        <div className="max-w-md mx-auto bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-bold text-center mb-4">Thanh toán bằng Solana</h2>
            
            <div className="text-center mb-6">
                <img 
                    src={paymentData.qrCode} 
                    alt="Solana Payment QR Code"
                    className="w-64 h-64 mx-auto border rounded-lg"
                />
            </div>

            <div className="bg-gray-50 p-4 rounded-lg mb-6">
                <div className="flex justify-between py-2">
                    <span>Đơn hàng:</span>
                    <span>#{orderId}</span>
                </div>
                <div className="flex justify-between py-2">
                    <span>Số tiền VND:</span>
                    <span>{paymentData.vndAmount.toLocaleString()} VND</span>
                </div>
                <div className="flex justify-between py-2">
                    <span>Số tiền SOL:</span>
                    <span>{paymentData.solAmount} SOL</span>
                </div>
                <div className="flex justify-between py-2 text-xs">
                    <span>Địa chỉ:</span>
                    <span className="font-mono break-all">{paymentData.recipient}</span>
                </div>
            </div>

            <div className="mb-6">
                <h4 className="font-bold mb-2">Hướng dẫn thanh toán:</h4>
                <ol className="text-sm text-gray-600 space-y-1">
                    <li>1. Mở ví Solana trên điện thoại (Phantom, Solflare, ...)</li>
                    <li>2. Quét mã QR phía trên</li>
                    <li>3. Xác nhận giao dịch trong ví</li>
                    <li>4. Copy Transaction Signature</li>
                    <li>5. Nhấn "Tôi đã thanh toán" và nhập signature</li>
                </ol>
            </div>

            <div className="flex gap-3">
                <button 
                    onClick={handlePaymentComplete}
                    disabled={loading}
                    className="flex-1 bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 disabled:opacity-50"
                >
                    {loading ? 'Đang xác minh...' : 'Tôi đã thanh toán'}
                </button>
                <button 
                    onClick={onPaymentCancel}
                    className="flex-1 bg-gray-600 text-white py-3 px-4 rounded-lg hover:bg-gray-700"
                >
                    Hủy
                </button>
            </div>
        </div>
    );
};

export default SolanaPayment;
