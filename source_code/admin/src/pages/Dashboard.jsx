import React, { useEffect, useState } from "react";
import axios from "axios";
import { backendUrl, currency } from "../App";
import { toast } from "react-toastify";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const Dashboard = ({ token }) => {
  const [revenue, setRevenue] = useState(0);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState("day"); // 'day', 'month', 'product'
  const [chartData, setChartData] = useState([]);
  const [year, setYear] = useState(new Date().getFullYear());
  const [from, setFrom] = useState(() =>
    new Date(new Date().setDate(new Date().getDate() - 30))
      .toISOString()
      .slice(0, 10)
  );
  const [to, setTo] = useState(() => new Date().toISOString().slice(0, 10));

  const fetchRevenue = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${backendUrl}/api/order/revenue`, {
        headers: { token },
      });
      if (response.data.success) {
        setRevenue(response.data.total);
      } else {
        toast.error(response.data.message || "Không thể lấy dữ liệu doanh thu");
      }
    } catch (error) {
      console.error("Error fetching revenue:", error);
      toast.error("Không thể kết nối đến máy chủ");
    } finally {
      setLoading(false);
    }
  };

  const fetchChartData = async () => {
    try {
      setLoading(true);
      let url = "";
      let toParam = to;
      if (mode === "day" || mode === "product") {
        const today = new Date();
        const toDate = new Date(to);
        if (
          toDate.getFullYear() === today.getFullYear() &&
          toDate.getMonth() === today.getMonth() &&
          toDate.getDate() === today.getDate()
        ) {
          // Nếu ngày cuối là hôm nay, tăng lên 1 ngày để lấy hết order trong hôm nay
          const nextDay = new Date(toDate);
          nextDay.setDate(nextDay.getDate() + 1);
          toParam = nextDay.toISOString().slice(0, 10);
        }
      }
      if (mode === "day") {
        url = `${backendUrl}/api/order/revenue-by-day?from=${from}&to=${toParam}`;
      } else if (mode === "month") {
        url = `${backendUrl}/api/order/revenue-by-month?year=${year}`;
      } else if (mode === "product") {
        url = `${backendUrl}/api/order/revenue-by-product?from=${from}&to=${toParam}`;
      }
      const response = await axios.get(url, { headers: { token } });
      if (response.data.success) {
        setChartData(response.data.data);
      } else {
        toast.error(response.data.message || "Không thể lấy dữ liệu biểu đồ");
      }
    } catch (error) {
      toast.error("Không thể lấy dữ liệu biểu đồ");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRevenue();
    const handleStorage = (e) => {
      if (e.key === "revenueNeedsUpdate") {
        fetchRevenue();
        fetchChartData();
      }
    };
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
    // eslint-disable-next-line
  }, [token]);

  useEffect(() => {
    fetchChartData();
    // eslint-disable-next-line
  }, [mode, from, to, year, token]);

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Dashboard</h2>
      <div className="bg-white rounded shadow p-6 w-full max-w-md mb-8">
        <div className="text-lg font-semibold text-gray-700 mb-2">
          Tổng doanh thu
        </div>
        {loading ? (
          <div className="text-gray-500">Đang tải...</div>
        ) : (
          <div className="text-3xl font-bold text-green-600">
            {revenue.toLocaleString()} {currency}
          </div>
        )}
      </div>
      <div className="mb-4 flex gap-2">
        <button
          className={`px-4 py-2 rounded ${
            mode === "day"
              ? "bg-blue-500 text-white"
              : "bg-gray-200 text-gray-700"
          }`}
          onClick={() => setMode("day")}
        >
          Theo ngày
        </button>
        <button
          className={`px-4 py-2 rounded ${
            mode === "month"
              ? "bg-blue-500 text-white"
              : "bg-gray-200 text-gray-700"
          }`}
          onClick={() => setMode("month")}
        >
          Theo tháng
        </button>
        <button
          className={`px-4 py-2 rounded ${
            mode === "product"
              ? "bg-blue-500 text-white"
              : "bg-gray-200 text-gray-700"
          }`}
          onClick={() => setMode("product")}
        >
          Theo sản phẩm
        </button>
      </div>
      {/* Bộ lọc thời gian */}
      {mode === "day" || mode === "product" ? (
        <div className="mb-4 flex gap-2 items-center">
          <label>Từ:</label>
          <input
            type="date"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
            className="border rounded px-2 py-1"
          />
          <label>Đến:</label>
          <input
            type="date"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            className="border rounded px-2 py-1"
          />
        </div>
      ) : mode === "month" ? (
        <div className="mb-4 flex gap-2 items-center">
          <label>Năm:</label>
          <input
            type="number"
            value={year}
            onChange={(e) => setYear(e.target.value)}
            className="border rounded px-2 py-1 w-24"
          />
        </div>
      ) : null}
      {/* Biểu đồ */}
      <div className="bg-white rounded shadow p-6 w-full max-w-3xl">
        {loading ? (
          <div className="text-gray-500">Đang tải biểu đồ...</div>
        ) : mode === "day" ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="_id" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="total"
                stroke="#8884d8"
                name="Doanh thu"
              />
            </LineChart>
          </ResponsiveContainer>
        ) : mode === "month" ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="_id" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="total" fill="#82ca9d" name="Doanh thu" />
            </BarChart>
          </ResponsiveContainer>
        ) : mode === "product" ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="_id" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="total" fill="#ffc658" name="Doanh thu" />
              <Bar dataKey="quantity" fill="#8884d8" name="Số lượng bán" />
            </BarChart>
          </ResponsiveContainer>
        ) : null}
      </div>
    </div>
  );
};

export default Dashboard;
