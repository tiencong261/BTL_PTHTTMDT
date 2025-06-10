import orderModel from "../models/orderModel.js";
import userModel from "../models/userModel.js";
import Stripe from "stripe";
// import Razorpay from "razorpay";
import moment from "moment";
import crypto from "crypto";
import querystring from "qs";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";
import mongoose from "mongoose";

// global variables
const currency = "inr"; // Lưu ý: VNPAY chỉ hỗ trợ VND
const deliveryCharge = 10;

// gateway initialize
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
// const razorpay = new Razorpay({
//   key_id: process.env.RAZORPAY_KEY_ID,
//   key_secret: process.env.RAZORPAY_KEY_SECRET,
// });

//Placing orders using COD Method
const placeOrder = async (req, res) => {
  try {
    const { userId, items, amount, address } = req.body;

    const orderData = {
      userId,
      items,
      address,
      amount,
      paymentMethod: "COD",
      payment: false,
      date: Date.now(),
    };

    const newOrder = new orderModel(orderData);
    await newOrder.save();

    await userModel.findByIdAndUpdate(userId, { cartData: {} });

    res.json({ success: true, message: "Order Placed" });
  } catch (error) {
    console.error(error);
    res.json({ success: false, message: error.message });
  }
};

//Placing orders using Stripe Method
const placeOrderStripe = async (req, res) => {
  try {
    const { userId, items, amount, address } = req.body;
    const { origin } = req.body;

    const orderData = {
      userId,
      items,
      address,
      amount,
      paymentMethod: "Stripe",
      payment: false,
      date: Date.now(),
    };

    const newOrder = new orderModel(orderData);
    await newOrder.save();

    const line_items = items.map((item) => ({
      price_data: {
        currency: currency,
        product_data: {
          name: item.name,
        },
        unit_amount: item.price * 100,
      },
      quantity: item.quantity,
    }));

    line_items.push({
      price_data: {
        currency: currency,
        product_data: {
          name: "Delivery Charges",
        },
        unit_amount: deliveryCharge * 100,
      },
      quantity: 1,
    });

    const session = await stripe.checkout.sessions.create({
      success_url: `${origin}/verify?success=true&orderId=${newOrder._id}`,
      cancel_url: `${origin}/verify?success=false&orderId=${newOrder._id}`,
      line_items,
      mode: "payment",
    });

    res.json({ success: true, session_url: session.url });
  } catch (error) {
    console.error(error);
    res.json({ success: false, message: error.message });
  }
};

//Placing orders using Razorpay Method
const placeOrderRazorpay = async (req, res) => {
  try {
    const { userId, items, amount, address } = req.body;

    const orderData = {
      userId,
      items,
      address,
      amount,
      paymentMethod: "Razorpay",
      payment: false,
      date: Date.now(),
    };

    const newOrder = new orderModel(orderData);
    await newOrder.save();

    const options = {
      amount: amount * 100, // Razorpay expects amount in paise
      currency: currency.toUpperCase(),
      receipt: newOrder._id.toString(),
    };

    const order = await razorpay.orders.create(options);

    res.json({
      success: true,
      order: {
        id: order.id,
        amount: order.amount,
        currency: order.currency,
        receipt: order.receipt,
      },
    });
  } catch (error) {
    console.error(error);
    res.json({ success: false, message: error.message });
  }
};

//Verify Razorpay Payment
const verifyRazorpay = async (req, res) => {
  try {
    const {
      orderId,
      razorpay_payment_id,
      razorpay_order_id,
      razorpay_signature,
    } = req.body;

    const body = razorpay_order_id + "|" + razorpay_payment_id;
    const expectedSignature = crypto
      .createHmac("sha256", process.env.RAZORPAY_KEY_SECRET)
      .update(body.toString())
      .digest("hex");

    if (expectedSignature === razorpay_signature) {
      await orderModel.findByIdAndUpdate(orderId, {
        payment: true,
        paymentTime: Date.now(),
      });
      await userModel.findByIdAndUpdate(req.body.userId, { cartData: {} });
      res.json({ success: true });
    } else {
      res.json({ success: false, message: "Invalid signature" });
    }
  } catch (error) {
    console.error(error);
    res.json({ success: false, message: error.message });
  }
};

// Verify Stripe
const verifyStripe = async (req, res) => {
  const { orderId, success, userId } = req.body;

  try {
    if (success === "true") {
      await orderModel.findByIdAndUpdate(orderId, { payment: true });
      await userModel.findByIdAndUpdate(userId, { cartData: {} });
      res.json({ success: true });
    } else {
      await orderModel.findByIdAndDelete(orderId);
      res.json({ success: false });
    }
  } catch (error) {
    console.error(error);
    res.json({ success: false, message: error.message });
  }
};

//Placing orders using VNPAY Method
const placeOrderVnpay = async (req, res) => {
  try {
    const { userId, items, amount, address } = req.body;
    // Ensure amount is integer and in VND
    const amountVND = Math.round(Number(amount));
    if (!amountVND || isNaN(amountVND) || amountVND <= 0) {
      return res
        .status(400)
        .json({ success: false, message: "Số tiền không hợp lệ" });
    }

    // Create new order in database
    const newOrder = new orderModel({
      userId,
      items,
      address,
      amount: amountVND,
      paymentMethod: "VNPay",
      payment: false,
      date: Date.now(),
    });
    await newOrder.save();

    // Get IP address
    const ipAddr =
      req.headers["x-forwarded-for"] || req.connection.remoteAddress;

    // Prepare VNPAY parameters
    const vnpParams = {
      vnp_Version: "2.1.0",
      vnp_Command: "pay",
      vnp_TmnCode: process.env.VNPAY_TMN_CODE,
      vnp_Amount: amountVND * 100, // VNPAY yêu cầu số nguyên, đơn vị là đồng, nhân 100
      vnp_CurrCode: "VND",
      vnp_TxnRef: newOrder._id.toString(),
      vnp_OrderInfo: `Thanh toan don hang #${newOrder._id}`,
      vnp_OrderType: "other",
      vnp_Locale: "vn",
      vnp_ReturnUrl: `${process.env.BACKEND_URL}/api/order/vnpay_return`, // Backend endpoint for hash verification
      vnp_IpAddr: ipAddr === "::1" ? "127.0.0.1" : ipAddr,
      vnp_CreateDate: moment().format("YYYYMMDDHHmmss"),
      vnp_ExpireDate: moment().add(15, "minutes").format("YYYYMMDDHHmmss"),
    };

    // Sort parameters alphabetically
    const sortedParams = sortObject(vnpParams);
    // Log for debugging
    console.log("VNPAY PARAMS:", sortedParams);

    // Create secure hash
    const signData = querystring.stringify(sortedParams, { encode: false });
    const hmac = crypto.createHmac("sha512", process.env.VNPAY_HASH_SECRET);
    const signed = hmac.update(Buffer.from(signData, "utf-8")).digest("hex");
    sortedParams["vnp_SecureHash"] = signed;

    // Create payment URL
    const paymentUrl = `${process.env.VNPAY_URL}?${querystring.stringify(
      sortedParams,
      {
        encode: false,
      }
    )}`;

    res.json({
      success: true,
      message: "Redirecting to VNPay",
      paymentUrl,
      orderId: newOrder._id,
    });
  } catch (err) {
    console.error("VNPay Error:", err);
    res.status(500).json({
      success: false,
      message: err.message || "Error creating VNPay payment",
    });
  }
};

// VNPAY Return Handler
const vnpayReturn = async (req, res) => {
  try {
    const query = req.query;
    const vnp_SecureHash = query.vnp_SecureHash;

    // Remove secure hash from query for signature verification
    delete query.vnp_SecureHash;
    delete query.vnp_SecureHashType;

    // Sort parameters and create signature
    const sortedQuery = sortObject(query);
    const signData = querystring.stringify(sortedQuery, { encode: false });
    const hmac = crypto.createHmac("sha512", process.env.VNPAY_HASH_SECRET);
    const signed = hmac.update(Buffer.from(signData, "utf-8")).digest("hex");

    // Verify signature
    if (signed === vnp_SecureHash) {
      const orderId = query.vnp_TxnRef;
      const responseCode = query.vnp_ResponseCode;

      if (responseCode === "00") {
        // Payment successful
        await orderModel.findByIdAndUpdate(orderId, {
          payment: true,
          paymentTime: Date.now(),
        });

        // Redirect to success page
        return res.redirect(
          `${process.env.FRONTEND_URL}/orders?payment=success`
        );
      } else {
        // Payment failed
        return res.redirect(
          `${process.env.FRONTEND_URL}/orders?payment=failed&code=${responseCode}`
        );
      }
    } else {
      // Invalid signature
      return res.redirect(`${process.env.FRONTEND_URL}/orders?payment=invalid`);
    }
  } catch (err) {
    console.error("VNPay Return Error:", err);
    return res.redirect(`${process.env.FRONTEND_URL}/orders?payment=error`);
  }
};

// Helper function to sort object keys alphabetically
function sortObject(obj) {
  const sorted = {};
  const keys = Object.keys(obj).sort();
  for (const key of keys) {
    sorted[key] = obj[key];
  }
  return sorted;
}

//Placing orders using MOMO Method
const placeOrderMomo = async (req, res) => {
  try {
    const { userId, items, amount, address } = req.body;
    const amountVND = amount;

    const newOrder = new orderModel({
      userId,
      items,
      address,
      amount: amountVND,
      paymentMethod: "MOMO",
      payment: false,
      date: Date.now(),
    });
    await newOrder.save();

    const partnerCode = process.env.MOMO_PARTNER_CODE;
    const accessKey = process.env.MOMO_ACCESS_KEY;
    const secretKey = process.env.MOMO_SECRET_KEY;
    const requestId = newOrder._id.toString();
    const orderId = newOrder._id.toString();
    const orderInfo = `Thanh toan don hang #${newOrder._id}`;
    const redirectUrl = `${req.protocol}://${req.get("host")}/orders`;
    const ipnUrl = `${req.protocol}://${req.get("host")}/api/order/momo_ipn`;
    const requestType = "captureWallet";
    const extraData = "";

    const rawSignature = `accessKey=${accessKey}&amount=${amountVND}&extraData=${extraData}&ipnUrl=${ipnUrl}&orderId=${orderId}&orderInfo=${orderInfo}&partnerCode=${partnerCode}&redirectUrl=${redirectUrl}&requestId=${requestId}&requestType=${requestType}`;
    const signature = crypto
      .createHmac("sha256", secretKey)
      .update(rawSignature)
      .digest("hex");

    const requestBody = {
      partnerCode,
      accessKey,
      requestId,
      amount: amountVND,
      orderId,
      orderInfo,
      redirectUrl,
      ipnUrl,
      extraData,
      requestType,
      signature,
      lang: "vi",
    };

    const response = await axios.post(
      "https://test-payment.momo.vn/v2/gateway/api/create",
      requestBody
    );

    if (response.data.resultCode === 0) {
      res.json({ success: true, paymentUrl: response.data.payUrl });
    } else {
      throw new Error(response.data.message);
    }
  } catch (err) {
    console.error("MOMO Error:", err);
    res.status(500).json({ success: false, message: err.message });
  }
};

//All order data for Admin Panel
const allOrder = async (req, res) => {
  try {
    const orders = await orderModel.find({});
    res.json({ success: true, orders });
  } catch (error) {
    console.error(error);
    res.json({ success: false, message: error.message });
  }
};

//User order data for Forntend
const userOrders = async (req, res) => {
  try {
    const { userId } = req.body;
    const orders = await orderModel.find({ userId });
    res.json({ success: true, orders });
  } catch (error) {
    console.error(error);
    res.json({ success: false, message: error.message });
  }
};

//update order status from Admin Panel
const updateStatus = async (req, res) => {
  try {
    const { orderId, status } = req.body;
    await orderModel.findByIdAndUpdate(orderId, { status });
    res.json({ success: true, message: "Status Updated" });
  } catch (error) {
    console.error(error);
    res.json({ success: false, message: error.message });
  }
};

const deleteAllOrders = async (req, res) => {
  try {
    const result = await orderModel.deleteMany({});
    console.log("Deleted orders:", result);
    res.json({ success: true, message: "All orders have been deleted" });
  } catch (error) {
    console.error("Error deleting all orders:", error);
    res.status(500).json({ success: false, message: "Error deleting orders" });
  }
};

export const getRevenue = async (req, res) => {
  try {
    const result = await orderModel.aggregate([
      { $match: { payment: true, refunded: { $ne: true } } },
      { $group: { _id: null, total: { $sum: "$amount" } } },
    ]);
    const total = result[0]?.total || 0;
    res.json({ success: true, total });
  } catch (error) {
    res
      .status(500)
      .json({ success: false, message: "Error calculating revenue" });
  }
};

// User gửi yêu cầu đổi trả/hoàn tiền
const requestReturnOrder = async (req, res) => {
  try {
    const { orderId, reason } = req.body;
    const order = await orderModel.findById(orderId);
    if (!order)
      return res
        .status(404)
        .json({ success: false, message: "Không tìm thấy đơn hàng" });
    if (order.returnRequest && order.returnRequest.requested) {
      return res.status(400).json({
        success: false,
        message: "Đơn hàng đã gửi yêu cầu đổi trả trước đó",
      });
    }
    order.returnRequest = {
      requested: true,
      reason,
      status: "pending",
      requestDate: new Date(),
    };
    await order.save();
    res.json({ success: true, message: "Đã gửi yêu cầu đổi trả/hoàn tiền" });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// Admin xử lý yêu cầu đổi trả/hoàn tiền
const processReturnOrder = async (req, res) => {
  try {
    const { orderId, status, adminNote } = req.body;
    const order = await orderModel.findById(orderId);
    if (!order || !order.returnRequest || !order.returnRequest.requested) {
      return res
        .status(404)
        .json({ success: false, message: "Không tìm thấy yêu cầu đổi trả" });
    }
    order.returnRequest.status = status;
    order.returnRequest.adminNote = adminNote;
    order.returnRequest.processDate = new Date();
    if (status === "approved" && order.payment === true) {
      order.refunded = true;
    }
    await order.save();
    res.json({
      success: true,
      message: `Đã cập nhật trạng thái đổi trả: ${status}`,
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// Lấy danh sách đơn hàng có yêu cầu đổi trả (cho admin)
const getReturnOrders = async (req, res) => {
  try {
    const orders = await orderModel.find({ "returnRequest.requested": true });
    res.json({ success: true, orders });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// Thống kê doanh thu theo ngày
const getRevenueByDay = async (req, res) => {
  try {
    const { from, to } = req.query;
    const match = {
      payment: true,
      refunded: { $ne: true },
      date: {
        $gte: new Date(from).getTime(),
        $lte: new Date(to).getTime(),
      },
    };
    const result = await orderModel.aggregate([
      { $match: match },
      {
        $group: {
          _id: {
            $dateToString: { format: "%Y-%m-%d", date: { $toDate: "$date" } },
          },
          total: { $sum: "$amount" },
        },
      },
      { $sort: { _id: 1 } },
    ]);
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// Thống kê doanh thu theo tháng
const getRevenueByMonth = async (req, res) => {
  try {
    const { year } = req.query;
    const start = new Date(`${year}-01-01`).getTime();
    const end = new Date(`${Number(year) + 1}-01-01`).getTime();
    const match = {
      payment: true,
      refunded: { $ne: true },
      date: { $gte: start, $lt: end },
    };
    const result = await orderModel.aggregate([
      { $match: match },
      {
        $group: {
          _id: {
            $dateToString: { format: "%Y-%m", date: { $toDate: "$date" } },
          },
          total: { $sum: "$amount" },
        },
      },
      { $sort: { _id: 1 } },
    ]);
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// Thống kê doanh thu theo sản phẩm
const getRevenueByProduct = async (req, res) => {
  try {
    const { from, to } = req.query;
    const match = {
      payment: true,
      refunded: { $ne: true },
      date: {
        $gte: new Date(from).getTime(),
        $lte: new Date(to).getTime(),
      },
    };
    const result = await orderModel.aggregate([
      { $match: match },
      { $unwind: "$items" },
      {
        $group: {
          _id: "$items.name",
          total: { $sum: { $multiply: ["$items.price", "$items.quantity"] } },
          quantity: { $sum: "$items.quantity" },
        },
      },
      { $sort: { total: -1 } },
    ]);
    res.json({ success: true, data: result });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

export {
  verifyStripe,
  placeOrder,
  placeOrderStripe,
  placeOrderRazorpay,
  verifyRazorpay,
  placeOrderVnpay,
  vnpayReturn,
  placeOrderMomo,
  allOrder,
  userOrders,
  updateStatus,
  deleteAllOrders,
  requestReturnOrder,
  processReturnOrder,
  getReturnOrders,
  getRevenueByDay,
  getRevenueByMonth,
  getRevenueByProduct,
};
