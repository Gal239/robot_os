# AI Orchestration Dashboard

A beautiful, interactive web dashboard for visualizing AI orchestration task graphs, logs, and agent activities.

## Features

✨ **Interactive Node Graph** - D3.js-powered visualization of task dependencies
📊 **Real-time Metrics** - Task statistics, agent activity, and timeline stats
📝 **Detailed Logs** - Master logs, metalogs, and timeline views
🔍 **Task Inspector** - Click any node to see full details, timeline, and relationships
🎨 **Modern UI** - Clean, professional design with smooth animations
📱 **Responsive** - Works on desktop, tablet, and mobile

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn
```

### 2. Run the Server

```bash
# From the project root
python -m ai_orchestration.agent_orc_web.server

# Or directly
cd ai_orchestration/agent_orc_web
python server.py
```

### 3. Open Dashboard

Navigate to [http://localhost:8000](http://localhost:8000) in your browser.

## Project Structure

```
ai_orchestration/web/
├── api.py                    # FastAPI backend
├── server.py                 # Server entry point
├── static/
│   ├── css/
│   │   ├── main.css         # Global styles
│   │   ├── navbar.css       # Navigation bar
│   │   ├── cards.css        # Metric cards
│   │   ├── graph.css        # Graph visualization
│   │   ├── sidebar.css      # Detail sidebar
│   │   ├── logs.css         # Logs section
│   │   └── animations.css   # Animations
│   └── js/
│       ├── main.js          # Main app logic
│       ├── api.js           # API communication
│       ├── cards.js         # Metric cards
│       ├── sidebar.js       # Detail sidebar
│       ├── graph.js         # D3.js graph
│       └── utils.js         # Utilities
└── templates/
    └── index.html           # Main HTML
```

## API Endpoints

- `GET /` - Dashboard homepage
- `GET /api/sessions` - List all sessions
- `GET /api/sessions/{session_id}` - Get session details
- `GET /api/sessions/{session_id}/graph` - Get graph structure
- `GET /api/sessions/{session_id}/logs` - Get logs
- `GET /api/tasks/{task_id}` - Get task details
- `GET /api/agents` - List agents
- `GET /api/health` - Health check

## Development

### Enable Auto-reload

The server runs with `reload=True` by default, so changes to Python files will automatically restart the server.

### Frontend Development

Edit files in `static/` and refresh your browser to see changes.

### Adding Custom Styles

Add your custom CSS to the appropriate file in `static/css/`.

### Extending the API

Add new endpoints in `api.py` and corresponding frontend logic in the JS files.

## Technologies

- **Backend**: FastAPI, Uvicorn, Python 3.12+
- **Frontend**: Vanilla JavaScript (ES6+), D3.js v7
- **Styling**: Custom CSS (no frameworks)
- **Icons**: Lucide Icons
- **Fonts**: Inter, JetBrains Mono (Google Fonts)

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Performance

- Virtual scrolling for large log lists
- Efficient D3.js rendering with viewport culling
- Client-side caching with TTL
- Debounced search and filters

## Troubleshooting

### Port Already in Use

```bash
# Change port in server.py or kill the process using port 8000
lsof -ti:8000 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :8000   # Windows
```

### Sessions Not Loading

- Check that your `runs/` directory exists and contains session data
- Verify `graph.json` files are valid JSON
- Check browser console for errors

### Graph Not Rendering

- Ensure D3.js is loaded (check browser console)
- Verify graph data has valid node and edge structures
- Try refreshing the page

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details
