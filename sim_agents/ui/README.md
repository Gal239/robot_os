# Echo Scene Maker UI

Beautiful web interface for chatting with Echo and building robot simulation scenes.

## Features

- 💬 **Chat with Echo** - Natural language scene creation
- 🎬 **Scene Viewer** - Multi-camera screenshots
- 📝 **Script Viewer** - Generated Python code
- 📊 **Scene Data** - Cameras, assets, sensors
- 📜 **Metalog** - Conversation context
- 🎨 **Pastel Design** - Clean, minimal, beautiful

## Tech Stack

**Backend:**
- Flask (Python web framework)
- Echo Scene Maker agent (AI orchestration)

**Frontend:**
- Tailwind CSS (styling, via CDN)
- Alpine.js (reactivity, via CDN)
- Animate.css (animations, via CDN)
- Component-based architecture (no build step!)

## Installation

1. **Install Python dependencies:**
```bash
cd core/sim_agents/ui
pip install -r requirements.txt
```

2. **That's it!** No frontend build needed - all libraries loaded via CDN.

## Running

**IMPORTANT:** Run from the `simulation_center` root directory!

1. **Navigate to simulation_center:**
```bash
cd /home/gal-labs/PycharmProjects/echo_robot/simulation_center
```

2. **Start the backend server:**
```bash
python core/sim_agents/ui/backend/server.py
```

3. **Open your browser:**
```
http://localhost:5050
```

## Usage

1. **Start chatting** - Type a message in the chat sidebar
2. **Ask Echo what he can do** - "What can you build?"
3. **Request a scene** - "Create a scene with a table and apple"
4. **View results** - Switch between Scene/Script/Data/Metalog tabs

## Architecture

```
ui/
├── backend/
│   ├── server.py          # Flask server
│   ├── api.py             # API endpoints
│   └── echo_ops.py        # Echo conversation manager
│
└── frontend/
    ├── index.html         # Entry point
    │
    ├── components/        # Small reusable pieces
    │   ├── message-bubble/
    │   ├── chat-input/
    │   ├── chat-header/
    │   └── tab-button/
    │
    ├── layouts/           # Composed sections
    │   ├── chat-sidebar/
    │   ├── app-bar/
    │   └── tabbed-area/
    │
    ├── pages/             # Full pages
    │   └── scene-maker/
    │
    ├── shared_scripts/
    │   ├── api.js
    │   └── utils.js
    │
    └── shared_styles/
        ├── main.css
        └── animations.css
```

## API Endpoints

- `POST /api/start` - Start conversation
- `POST /api/chat` - Send message to Echo
- `GET /api/conversation` - Get conversation history
- `GET /api/screenshots` - Get scene screenshots
- `GET /api/scene-script` - Get scene script
- `GET /api/scene-data` - Get scene metadata
- `GET /api/metalog` - Get conversation context
- `GET /api/status` - Get Echo's status

## Development

No build step required! Edit files and refresh browser.

**Hot tips:**
- Open browser DevTools (F12) to see console logs
- Check Network tab to debug API calls
- Use Chrome/Firefox for best results

## Troubleshooting

**Backend won't start:**
- Check Python dependencies: `pip install -r requirements.txt`
- Check port 5050 is available

**Frontend not loading:**
- Check browser console for errors
- Verify CDN libraries are loading (check Network tab)
- Try hard refresh (Ctrl+Shift+R)

**Echo not responding:**
- Check backend logs for errors
- Verify orchestrator is initialized correctly
- Check Claude API key is configured

## Future Enhancements

- [ ] WebSocket for real-time updates
- [ ] Save/load conversation sessions
- [ ] Export scenes to file
- [ ] Dark mode toggle
- [ ] Mobile responsive improvements

---

Built with ❤️  by Vibe Robotics
