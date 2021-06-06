from django.shortcuts import render

from django.http import HttpResponse, JsonResponse
from matplotlib import pylab
from pylab import *
import PIL, PIL.Image
from io import BytesIO
import base64

# Create your views here.

def index(request):
    return render(request, './index.html')
 
def showimage(request):
    # Construct the graph
    t = arange(0.0, 2.0, 0.01)
    s = sin(2*pi*t)
    plot(t, s, linewidth=1.0)
 
    xlabel('time (s)')
    ylabel('voltage (mV)')
    title('About as simple as it gets, folks')
    grid(True)
 
    # Store image in a string buffer
    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
 
    # Send buffer in a http response the the browser with the mime type image/png set
    return HttpResponse(buffer.getvalue(), content_type="image/png")

def upload_csv(request):
    data = {}
    csv_file = request.FILES['csv_file']
    if not csv_file.name.endswith('.csv'):
        pass
        # error
    file_data = csv_file.read().decode('utf-8')

    lines = file_data.split('\n')
    
    for line in lines:
        fields = line.split(',')
        data_dict = {}
        data_dict['Gender'] = fields[0]
        data_dict['Age'] = fields[1]
        data_dict['Annual Income'] = fields[2]
        data_dict['Spending Score'] = fields[3]
        
        form = EventsForm(data_dict)
        form.save()

    context = {
        'data': data
    }
    
    return render(request, './csv.html', context)