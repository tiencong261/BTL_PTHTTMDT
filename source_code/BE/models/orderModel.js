import mongoose from "mongoose";

const orderSchema = new mongoose.Schema({
  userId: { type: String, required: true },
  items: { type: Array, required: true },
  amount: { type: Number, required: true },
  address: { type: Object, required: true },
  status: { type: String, required: true, default: "Order Place" },
  paymentMethod: { type: String, required: true },
  payment: { type: Boolean, required: true, default: false },
  date: { type: Number, required: true },
  returnRequest: {
    type: {
      requested: { type: Boolean, default: false },
      reason: { type: String },
      status: {
        type: String,
        enum: ["pending", "approved", "rejected", "refunded"],
        default: "pending",
      },
      adminNote: { type: String },
      requestDate: { type: Date },
      processDate: { type: Date },
    },
    default: undefined,
  },
  refunded: { type: Boolean, default: false },
});

const orderModel = mongoose.model.order || mongoose.model("order", orderSchema);
export default orderModel;
