import express from "express";
import {
  placeOrder,
  placeOrderStripe,
  placeOrderRazorpay,
  placeOrderVnpay,
  placeOrderMomo,
  allOrder,
  userOrders,
  updateStatus,
  verifyStripe,
  deleteAllOrders,
  getRevenue,
  vnpayReturn,
  verifyRazorpay,
  requestReturnOrder,
  processReturnOrder,
  getReturnOrders,
  getRevenueByDay,
  getRevenueByMonth,
  getRevenueByProduct,
  placeOrderSolana,
  verifySolana,
} from "../controllers/orderController.js";
import adminAuth from "../middleware/adminAuth.js";
import authUser from "../middleware/auth.js";

const orderRouter = express.Router();

//Admin Features
orderRouter.post("/list", adminAuth, allOrder);
orderRouter.post("/status", adminAuth, updateStatus);
orderRouter.post("/delete-all", adminAuth, deleteAllOrders);
orderRouter.get("/revenue", adminAuth, getRevenue);

// Thống kê doanh thu
orderRouter.get("/revenue-by-day", getRevenueByDay);
orderRouter.get("/revenue-by-month", getRevenueByMonth);
orderRouter.get("/revenue-by-product", getRevenueByProduct);

//Payment Features
orderRouter.post("/solana", authUser, placeOrderSolana);
orderRouter.post("/verifySolana", authUser, verifySolana);
orderRouter.post("/place", authUser, placeOrder);
orderRouter.post("/stripe", authUser, placeOrderStripe);
orderRouter.post("/razorpay", authUser, placeOrderRazorpay);
orderRouter.post("/vnpay", authUser, placeOrderVnpay);
orderRouter.post("/momo", authUser, placeOrderMomo);
orderRouter.get("/vnpay_return", vnpayReturn);
orderRouter.post("/verifyRazorpay", authUser, verifyRazorpay);

//User Features
orderRouter.post("/userorders", authUser, userOrders);

// verify payment
orderRouter.post("/verifyStripe", authUser, verifyStripe);

// Đổi trả/hoàn tiền
orderRouter.post("/return-request", authUser, requestReturnOrder); // user gửi yêu cầu
orderRouter.post("/process-return", adminAuth, processReturnOrder); // admin xử lý
orderRouter.get("/return-orders", adminAuth, getReturnOrders); // admin lấy danh sách

export default orderRouter;
