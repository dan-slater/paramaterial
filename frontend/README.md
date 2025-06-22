# ParaMaterial Frontend

A modern React + TypeScript frontend for the ParaMaterial platform, built with Vite and powered by Zustand for state management.

## 🏗️ Architecture

### State Management with Zustand

The frontend uses **Zustand** for lightweight, type-safe state management:

- **🔐 Auth Store** (`authStore.ts`) - User authentication, login/logout, session management
- **📋 Jobs Store** (`jobsStore.ts`) - Job creation, fetching, status polling, file uploads
- **🏢 Organizations Store** (`organizationsStore.ts`) - Organization management (ready for API integration)
- **🎨 UI Store** (`uiStore.ts`) - Notifications, theme, sidebar state, loading indicators

### Key Features

- **Type-safe state management** with full TypeScript support
- **Persistent auth state** with automatic token refresh
- **Real-time notifications** with auto-dismiss and custom animations
- **Optimistic updates** for better UX
- **Background polling** for job status updates
- **Centralized error handling** with user-friendly notifications

## 🚀 Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Header.tsx
│   ├── JobCard.tsx
│   ├── FileUpload.tsx
│   └── NotificationContainer.tsx
├── pages/              # Route components
│   ├── Dashboard.tsx
│   ├── Login.tsx
│   ├── Register.tsx
│   ├── Upload.tsx
│   └── JobDetails.tsx
├── stores/             # Zustand stores
│   ├── authStore.ts    # Authentication state
│   ├── jobsStore.ts    # Jobs management
│   ├── organizationsStore.ts
│   ├── uiStore.ts      # UI state & notifications
│   └── index.ts        # Store exports
├── providers/          # React providers
│   └── AuthProvider.tsx
├── types/              # TypeScript type definitions
│   └── index.ts
├── utils/              # Utilities and API client
│   └── api.ts
└── App.tsx             # Main app component
```

## 🔧 Store Usage

### Auth Store

```typescript
import { useAuthStore } from './stores/authStore';

function LoginComponent() {
  const { login, user, loading, error } = useAuthStore();
  
  const handleLogin = async (credentials) => {
    await login(credentials);
    // User is automatically updated on success
  };
}
```

### Jobs Store

```typescript
import { useJobsStore } from './stores/jobsStore';

function JobsList() {
  const { jobs, fetchJobs, createJob, loading } = useJobsStore();
  
  useEffect(() => {
    fetchJobs();
  }, []);
}
```

### UI Store (Notifications)

```typescript
import { useUIStore } from './stores/uiStore';

function SomeComponent() {
  const { addNotification } = useUIStore();
  
  const showSuccess = () => {
    addNotification({
      type: 'success',
      title: 'Success!',
      message: 'Operation completed successfully',
      duration: 5000 // Optional, defaults to 5s
    });
  };
}
```

## 🎨 Styling

- **Tailwind CSS** for utility-first styling
- **Plasma colormap** inspired design tokens
- **Custom animations** for notifications and micro-interactions
- **Responsive design** with mobile-first approach

## 🔐 Authentication Flow

1. **Login/Register** → Updates auth store → Persists to localStorage
2. **Route protection** → Checks auth store state → Redirects if needed
3. **Token refresh** → Automatic handling via API interceptors
4. **Logout** → Clears auth store → Removes tokens → Redirects

## 📡 API Integration

The frontend communicates with the FastAPI backend through:

- **Axios client** with automatic token attachment
- **Request/response interceptors** for auth handling
- **Type-safe API functions** in `utils/api.ts`
- **Background polling** for real-time updates

## 🧪 Development

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint for code quality

### Environment Variables

```bash
VITE_API_URL=http://localhost:8000  # Backend API URL
```

## 🔄 Migration from Context API

This project was migrated from React Context API to Zustand for:

- **Better performance** - Selective re-renders based on used state
- **Simpler code** - Less boilerplate than Context + useReducer
- **DevTools support** - Built-in Redux DevTools integration
- **Persistence** - Easy state persistence with middleware
- **TypeScript support** - Better type inference and safety

## 🚀 Production Deployment

The frontend can be deployed to any static hosting service:

- **Vercel** (recommended)
- **Netlify**
- **AWS S3 + CloudFront**
- **Any CDN or web server**

## 🤝 Contributing

1. Follow the existing code style
2. Use TypeScript for all new code
3. Add proper type definitions
4. Update stores for new features
5. Test on multiple screen sizes
6. Update this README for new patterns

---

Built with ❤️ using React, TypeScript, Zustand, and Tailwind CSS.