import React, { useContext, useEffect, useState } from "react";
import { ShopContext } from "../context/ShopContext";
import {
  FaUser,
  FaEnvelope,
  FaPhone,
  FaMapMarkerAlt,
  FaEdit,
} from "react-icons/fa";
import axios from "axios";
import { toast } from "react-toastify";

const Profile = () => {
  const { user, token, backendUrl, setUser } = useContext(ShopContext);
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    phone: "",
    address: "",
  });

  useEffect(() => {
    if (user) {
      setFormData({
        name: user.name || "",
        email: user.email || "",
        phone: user.phone || "",
        address: user.address || "",
      });
    }
  }, [user]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.put(
        backendUrl + "/api/user/update",
        formData,
        { headers: { token } }
      );

      if (response.data.success) {
        setUser(response.data.user);
        setIsEditing(false);
        toast.success("Profile updated successfully!");
      } else {
        toast.error(response.data.message);
      }
    } catch (error) {
      console.error(error);
      toast.error("Failed to update profile");
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-600">Please login to view your profile</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white shadow-xl rounded-lg overflow-hidden">
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold text-white">
                Thông tin cá nhân
              </h2>
              <button
                onClick={() => setIsEditing(!isEditing)}
                className="text-white hover:text-gray-300 transition-colors"
              >
                <FaEdit className="w-5 h-5" />
              </button>
            </div>
          </div>

          <div className="p-6">
            {isEditing ? (
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <FaUser className="text-gray-500 w-5 h-5" />
                    <input
                      type="text"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      className="flex-1 p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-gray-500"
                      placeholder="Full Name"
                    />
                  </div>

                  <div className="flex items-center space-x-3">
                    <FaEnvelope className="text-gray-500 w-5 h-5" />
                    <input
                      type="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      className="flex-1 p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-gray-500"
                      placeholder="Email"
                    />
                  </div>

                  <div className="flex items-center space-x-3">
                    <FaPhone className="text-gray-500 w-5 h-5" />
                    <input
                      type="tel"
                      name="phone"
                      value={formData.phone}
                      onChange={handleInputChange}
                      className="flex-1 p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-gray-500"
                      placeholder="Phone Number"
                    />
                  </div>

                  <div className="flex items-center space-x-3">
                    <FaMapMarkerAlt className="text-gray-500 w-5 h-5" />
                    <input
                      type="text"
                      name="address"
                      value={formData.address}
                      onChange={handleInputChange}
                      className="flex-1 p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-gray-500"
                      placeholder="Address"
                    />
                  </div>
                </div>

                <div className="flex justify-end space-x-4">
                  <button
                    type="button"
                    onClick={() => setIsEditing(false)}
                    className="px-4 py-2 text-gray-600 hover:text-gray-800"
                  >
                    Hủy
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-700 transition-colors"
                  >
                    Lưu thay đổi
                  </button>
                </div>
              </form>
            ) : (
              <div className="space-y-6">
                <div className="flex items-center space-x-3">
                  <FaUser className="text-gray-500 w-5 h-5" />
                  <div>
                    <p className="text-sm text-gray-500">Họ tên</p>
                    <p className="text-lg">{user.name}</p>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <FaEnvelope className="text-gray-500 w-5 h-5" />
                  <div>
                    <p className="text-sm text-gray-500">Email</p>
                    <p className="text-lg">{user.email}</p>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <FaPhone className="text-gray-500 w-5 h-5" />
                  <div>
                    <p className="text-sm text-gray-500">Số điện thoại</p>
                    <p className="text-lg">{user.phone || "Not provided"}</p>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <FaMapMarkerAlt className="text-gray-500 w-5 h-5" />
                  <div>
                    <p className="text-sm text-gray-500">Địa chỉ</p>
                    <p className="text-lg">{user.address || "Not provided"}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
