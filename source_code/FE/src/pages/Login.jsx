
import React, { useContext, useEffect, useState } from 'react'
import { ShopContext } from '../context/ShopContext';
import axios from 'axios';
import { toast } from 'react-toastify';
import { FaEnvelope, FaLock, FaUser } from 'react-icons/fa';

const Login = () => {
  const [currencyState, setCurrencyState] = useState('Login');
  const { token, setToken, navigate, backendUrl } = useContext(ShopContext);

  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');

  const onSubmitHandle = async (event) => {
    event.preventDefault();
    try {
      if (currencyState === 'Sign Up') {
        const response = await axios.post(backendUrl + '/api/user/register', { name, email, password })
        if (response.data.success) {
          setToken(response.data.token);
          localStorage.setItem('token', response.data.token);
        } else {
          toast.error(response.data.message);
        }
      } else {
        const response = await axios.post(backendUrl + "/api/user/login", { email, password });
        if (response.data.success) {
          setToken(response.data.token);
          localStorage.setItem('token', response.data.token);
        } else {
          toast.error(response.data.message);
        }
      }
    } catch (error) {
      console.log(error);
      toast.error(error.message);
    }
  }

  useEffect(() => {
    if (token) {
      navigate('/');
    }
  }, [token]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-100 to-gray-300 px-4">
      <form onSubmit={onSubmitHandle} className="w-full max-w-md bg-white/60 backdrop-blur-md p-8 rounded-xl shadow-lg space-y-5">
        <h2 className="text-3xl text-center font-semibold text-gray-800 mb-4">{currencyState}</h2>

        {currencyState === 'Sign Up' && (
          <div className="flex items-center border border-gray-400 rounded-md px-3 py-2 bg-white">
            <FaUser className="text-gray-600 mr-2" />
            <input
              type="text"
              placeholder="Tên"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-transparent outline-none"
            />
          </div>
        )}

        <div className="flex items-center border border-gray-400 rounded-md px-3 py-2 bg-white">
          <FaEnvelope className="text-gray-600 mr-2" />
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full bg-transparent outline-none"
          />
        </div>

        <div className="flex items-center border border-gray-400 rounded-md px-3 py-2 bg-white">
          <FaLock className="text-gray-600 mr-2" />
          <input
            type="password"
            placeholder="Mật khẩu"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-transparent outline-none"
          />
        </div>

        <div className="flex justify-between text-sm text-gray-700">
          <p className="cursor-pointer hover:underline">Quên mật khẩu?</p>
          {currencyState === 'Login' ? (
            <p onClick={() => setCurrencyState('Sign Up')} className="cursor-pointer hover:underline">
              Đăng ký tài khoản
            </p>
          ) : (
            <p onClick={() => setCurrencyState('Login')} className="cursor-pointer hover:underline">
              Đăng nhập
            </p>
          )}
        </div>

        <button className="w-full py-2 bg-black text-white rounded-md hover:bg-gray-900 transition">
          {currencyState === 'Login' ? 'Sign In' : 'Sign Up'}
        </button>
      </form>
    </div>
  );
}

export default Login;
