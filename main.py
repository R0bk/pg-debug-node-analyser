import pickle
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')


templates = Jinja2Templates(directory='templates')


with open('query_data.pickle', 'rb') as handle:
    query_data = pickle.load(handle)

qd = {'%s_%s' % (log_id, entry_id): q_str for log_id, entry_id, _, q_str, _ in query_data}


@app.get('/')
def get_listing(request: Request):
    return templates.TemplateResponse(
        'listing.html',
        {'request': request, 'queries': query_data}
    )

@app.get('/plot/{plot_name}')
def get_plot(request: Request, plot_name):
    return templates.TemplateResponse(
        'plot.html',
        {'request': request, 'plot_name': plot_name, 'query': qd[plot_name]}
    )
