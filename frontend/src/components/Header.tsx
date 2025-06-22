import { useNavigate } from 'react-router-dom';
import { User, Bell, ChevronDown, LogOut, Settings } from 'lucide-react';
import { useAuthStore } from '../stores/authStore';
import { useUIStore } from '../stores/uiStore';

export function Header() {
  const navigate = useNavigate();
  const { user, logout, isAuthenticated } = useAuthStore();
  const { addNotification } = useUIStore();

  const handleLogout = async () => {
    await logout();
    addNotification({
      type: 'info',
      title: 'Signed out',
      message: 'You have been successfully signed out.',
    });
    navigate('/login');
  };

  return (
    <header className="w-full py-4 px-6 md:px-12 bg-white border-b border-gray-200 text-gray-900 flex items-center justify-between">
      <div className="flex items-center">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-plasma-gradient rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">PM</span>
          </div>
          <span className="text-xl font-bold text-plasma-gradient">ParaMaterial</span>
        </div>
      </div>

      {isAuthenticated && (
        <>
          <nav className="hidden md:flex items-center space-x-8 text-sm font-medium">
            <button 
              onClick={() => navigate('/dashboard')}
              className="hover:text-plasma-600 transition-colors"
            >
              Dashboard
            </button>
            <button 
              onClick={() => navigate('/upload')}
              className="hover:text-plasma-600 transition-colors"
            >
              Upload
            </button>
            <button 
              onClick={() => navigate('/jobs')}
              className="hover:text-plasma-600 transition-colors"
            >
              Jobs
            </button>
            <button 
              onClick={() => navigate('/organization')}
              className="hover:text-plasma-600 transition-colors"
            >
              Organization
            </button>
          </nav>

          <div className="flex items-center space-x-4">
            <button className="p-2 rounded-full hover:bg-gray-100 transition-colors">
              <Bell size={18} />
            </button>
            
            <div className="relative group">
              <div className="flex items-center space-x-2 cursor-pointer p-2 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="bg-plasma-500 rounded-full p-1">
                  <User size={18} className="text-white" />
                </div>
                <span className="hidden md:inline text-sm">
                  {user?.first_name || 'User'}
                </span>
                <ChevronDown size={16} />
              </div>

              {/* Dropdown menu */}
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-all duration-200 z-50">
                <div className="py-2">
                  <div className="px-4 py-2 border-b border-gray-100">
                    <p className="text-sm font-medium text-gray-900">
                      {user?.first_name} {user?.last_name}
                    </p>
                    <p className="text-xs text-gray-500">{user?.email}</p>
                  </div>
                  
                  <button className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center space-x-2">
                    <Settings size={16} />
                    <span>Settings</span>
                  </button>
                  
                  <button 
                    onClick={handleLogout}
                    className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center space-x-2"
                  >
                    <LogOut size={16} />
                    <span>Sign out</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {!isAuthenticated && (
        <div className="flex items-center space-x-4">
          <button 
            onClick={() => navigate('/login')}
            className="text-sm font-medium hover:text-plasma-600 transition-colors"
          >
            Sign in
          </button>
          <button 
            onClick={() => navigate('/register')}
            className="px-4 py-2 bg-plasma-500 text-white text-sm font-medium rounded-lg hover:bg-plasma-600 transition-colors"
          >
            Get started
          </button>
        </div>
      )}
    </header>
  );
}