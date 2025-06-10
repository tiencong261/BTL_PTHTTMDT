import React from 'react'
import { Route, Routes } from 'react-router-dom'
import Home from './pages/Home'
import About from './pages/About'
import Cart from './pages/Cart'
import Collection from './pages/Collection'
import Contact from './pages/Contact'
import Login from './pages/Login'
import Order from './pages/Order'
import PlaceOrder from './pages/PlaceOrder'
import Product from './pages/Product'
import Chatbot from './pages/Chatbot'
import Profile from './pages/Profile'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import SearchBar from './components/SearchBar'
import {toast,ToastContainer} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Verify from './pages/Verify'
import { useLocation } from 'react-router-dom';


const App = () => {
  const location = useLocation();
  const isLoginPage = location.pathname === '/login';

  return (
    <div className='px4 sm:px-[5vw] md:px-[7vw] lg:px-[9vw]'>
      <ToastContainer />
      {!isLoginPage && <Navbar />}
      {!isLoginPage && <SearchBar />}
      
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/collection' element={<Collection />} />
        <Route path='/about' element={<About />} />
        <Route path='/cart' element={<Cart />} />
        <Route path='/contact' element={<Contact />} />
        <Route path='/login' element={<Login />} />
        <Route path='/orders' element={<Order />} />
        <Route path='/chatbot' element={<Chatbot />} />
        <Route path='/place-order' element={<PlaceOrder />} />
        <Route path='/product/:productId' element={<Product />} />
        <Route path='/verify' element={<Verify />} />
        <Route path='profile' element={<Profile />} />
      </Routes>

      {!isLoginPage && <Footer />}
    </div>
  );
};
export default App