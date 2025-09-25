function varargout = sinesum2(varargin)
% SINESUM2 M-file for sinesum2.fig
%      SINESUM2, by itself, creates a new SINESUM2 or raises the existing
%      singleton*.
%
%      H = SINESUM2 returns the handle to a new SINESUM2 or the handle to
%      the existing singleton*.
%
%      SINESUM2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SINESUM2.M with the given input arguments.
%
%      SINESUM2('Property','Value',...) creates a new SINESUM2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before sinesum2_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to sinesum2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help sinesum2

% Last Modified by GUIDE v2.5 18-Nov-2006 15:53:53

% ---- Michelle Daniels, EE261, Autumn 2006 (based on sinesum) ----

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @sinesum2_OpeningFcn, ...
                   'gui_OutputFcn',  @sinesum2_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before sinesum2 is made visible.
function sinesum2_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to sinesum2 (see VARARGIN)

% Choose default command line output for sinesum2
handles.output = hObject;

% init data structures
handles.num_harmonics = 1;

% Update handles structure
guidata(hObject, handles);

% set GUI title
set(hObject, 'Name', 'Sum of Sines');

% initialize
init(hObject, handles);

% UIWAIT makes sinesum2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = sinesum2_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- INIT Initialize everything from handles.num_harmonics
function init(hObject, handles)
% hObject    handle to figure
% handles    structure with handles and user data (see GUIDATA)

% resize data structures
handles.amplitudes = zeros(handles.num_harmonics, 1);
handles.phases = zeros(handles.num_harmonics, 1);

set(findobj('Tag', 'num_harmonics_edit'), 'String', handles.num_harmonics);

% initialize harmonic pop-up menu
harmonics = cell(handles.num_harmonics, 1);
for n=1:handles.num_harmonics;
    harmonics{n} = n;
end;
set(findobj('Tag', 'current_harmonic_popupmenu'), 'String', harmonics);

% set current harmonic to be adjusted
handles.current_harmonic = 1;
guidata(hObject, handles);
set_current_harmonic(hObject, handles);

% --- UPDATE DISPLAY
function update_display(hObject, handles)
% hObject    handle to figure
% handles    structure with handles and user data (see GUIDATA)

% -- update listbox
listbox_handle = findobj('Tag', 'harmonics_listbox');
options = cell(handles.num_harmonics + 1, 1);
options{1} = 'Harmonic    Amplitude    Phase';
for n=1:handles.num_harmonics;
    options{n+1} = sprintf('%3.0d               %2.4f        %2.4f', n, handles.amplitudes(n,1), handles.phases(n,1));
end; 
set(listbox_handle, 'String', options);
set(listbox_handle, 'Value', handles.current_harmonic + 1);

maxT = 2;
intervalT = 1/200;
t=0:intervalT:maxT;

% -- plot combined signal
axes(handles.combined_signal_axes);
cla;
x = zeros(1, length(t));
for n=1:handles.num_harmonics
    x = x + handles.amplitudes(n,1) * sin(2*pi*n*t + handles.phases(n,1)); 
end
plot(t, x);
grid on;
maxX = max(max(max(abs(x)), 0.0001), handles.amplitudes(handles.current_harmonic,1));
axis([0 maxT -maxX maxX]);
title('Combined Signal');

% -- plot current harmonic
axes(handles.current_harmonic_axes);
cla;
y = handles.amplitudes(handles.current_harmonic,1) * sin(2*pi*handles.current_harmonic*t + handles.phases(handles.current_harmonic,1)); 
plot(t, y);
grid on;
axis([0 maxT -maxX maxX]);
title(sprintf('Harmonic %d', handles.current_harmonic));

% -- plot spectral profile
axes(handles.spectral_profile_axes);
cla;
stem3(1:handles.num_harmonics, handles.phases, handles.amplitudes,'fill');
hold on;
stem3(handles.current_harmonic, handles.phases(handles.current_harmonic,1), handles.amplitudes(handles.current_harmonic,1),'fill', '-r');
hold off;
axis([0 (handles.num_harmonics + 1) 0 2*pi 0 (max(handles.amplitudes) + 1)]);
title('Spectral Profile');
xlabel('Harmonic');
ylabel('Phase');
zlabel('Amplitude');


% --- SET CURRENT HARMONIC
function set_current_harmonic(hObject, handles)
% hObject    handle to figure
% handles    structure with handles and user data (see GUIDATA)
% update all controls with data for current harmonic
set(findobj('Tag', 'amplitude_edit'), 'String', num2str(handles.amplitudes(handles.current_harmonic, 1)));
set(findobj('Tag', 'phase_edit'), 'String', num2str(handles.phases(handles.current_harmonic, 1)));
set(findobj('Tag', 'amplitude_slider'), 'Value', handles.amplitudes(handles.current_harmonic, 1));
set(findobj('Tag', 'phase_slider'), 'Value', handles.phases(handles.current_harmonic, 1));
if (handles.current_harmonic == 1) set(findobj('Tag', 'previous_harmonic_pushbutton'), 'Enable', 'off'); 
else set(findobj('Tag', 'previous_harmonic_pushbutton'), 'Enable', 'on'); 
end
if (handles.current_harmonic == handles.num_harmonics) set(findobj('Tag', 'next_harmonic_pushbutton'), 'Enable', 'off'); 
else set(findobj('Tag', 'next_harmonic_pushbutton'), 'Enable', 'on'); 
end
set(findobj('Tag', 'current_harmonic_popupmenu'), 'Value', handles.current_harmonic);
update_display(hObject, handles);

% --- CALLBACKS ---

% --- Executes on button press in start_over_pushbutton.
function start_over_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to start_over_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.num_harmonics = max(1, floor(str2double(get(findobj('Tag', 'num_harmonics_edit'), 'String'))));
init(hObject, handles);

% --------------------------------------------------------------------
function FileMenu_Callback(hObject, eventdata, handles)
% hObject    handle to FileMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function OpenMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to OpenMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = uigetfile('*.mat');
if ~isequal(file, 0)
    data = load(file);
    fields = fieldnames(data);
    % - check for valid amplitude and phase info
    amp_found = false;
    phase_found = false;
    for i=1:length(fields)
        if (strcmp(fields{i}, 'Amplitudes'))
            amp_found = true;
        end
        if (strcmp(fields{i}, 'Phases'))
            phase_found = true;
        end
    end
    if (~amp_found || ~phase_found)
        errordlg('Valid amplitude and phase information not found in file.');
        return;
    end;
    amplitudes = data.Amplitudes;
    phases = data.Phases;
    if (length(amplitudes) ~= length(phases))
        errordlg('Valid amplitude and phase information not found in file.');
        return;
    end;     
    % - first initialize with new number of harmonics
    handles.num_harmonics = length(amplitudes);
    guidata(hObject, handles);
    init(hObject, handles);
    handles = guidata(hObject);
    % - then update display with stored amplitude and phase info
    handles.amplitudes = amplitudes;
    handles.phases = phases;
    guidata(hObject, handles);
    update_display(hObject, handles);
end

% --------------------------------------------------------------------
function SaveMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to SaveMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
file = uiputfile('*.mat', 'Save Current Project As');
if ~isequal(file, 0)
    Amplitudes = handles.amplitudes;
    Phases = handles.phases;
    save(file, 'Amplitudes', 'Phases', '-v6');
end

% --- Executes on selection change in current_harmonic_popupmenu.
function current_harmonic_popupmenu_Callback(hObject, eventdata, handles)
% hObject    handle to current_harmonic_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns current_harmonic_popupmenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from current_harmonic_popupmenu
popup_sel_index = get(handles.current_harmonic_popupmenu, 'Value');
handles.current_harmonic = popup_sel_index; % 1-indexed
guidata(hObject, handles);
set_current_harmonic(hObject, handles);

% --- Executes on selection change in harmonics_listbox.
function harmonics_listbox_Callback(hObject, eventdata, handles)
% hObject    handle to harmonics_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns harmonics_listbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from harmonics_listbox
handles.current_harmonic = max(1, get(hObject, 'Value') - 1);
guidata(hObject, handles);
set_current_harmonic(hObject, handles);

% --- Executes on slider movement.
function amplitude_slider_Callback(hObject, eventdata, handles)
% hObject    handle to amplitude_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
handles.amplitudes(handles.current_harmonic, 1) = get(hObject,'Value');
guidata(hObject, handles);
set(findobj('Tag', 'amplitude_edit'), 'String', num2str(handles.amplitudes(handles.current_harmonic, 1)));
update_display(hObject, handles);

% --- Executes on slider movement.
function phase_slider_Callback(hObject, eventdata, handles)
% hObject    handle to phase_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
handles.phases(handles.current_harmonic, 1) = get(hObject,'Value');
guidata(hObject, handles);
set(findobj('Tag', 'phase_edit'), 'String', num2str(handles.phases(handles.current_harmonic, 1)));
update_display(hObject, handles);

function amplitude_edit_Callback(hObject, eventdata, handles)
% hObject    handle to amplitude_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of amplitude_edit as text
%        str2double(get(hObject,'String')) returns contents of amplitude_edit as a double
handles.amplitudes(handles.current_harmonic, 1) = str2double(get(hObject,'String'));
guidata(hObject, handles);
set(findobj('Tag', 'amplitude_slider'), 'Value', handles.amplitudes(handles.current_harmonic, 1));
update_display(hObject, handles);

function phase_edit_Callback(hObject, eventdata, handles)
% hObject    handle to phase_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of phase_edit as text
%        str2double(get(hObject,'String')) returns contents of phase_edit as a double
handles.phases(handles.current_harmonic, 1) = str2double(get(hObject,'String'));
guidata(hObject, handles);
set(findobj('Tag', 'phase_slider'), 'Value', handles.phases(handles.current_harmonic, 1));
update_display(hObject, handles);

function num_harmonics_edit_Callback(hObject, eventdata, handles)
% hObject    handle to num_harmonics_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of num_harmonics_edit as text
%        str2double(get(hObject,'String')) returns contents of num_harmonics_edit as a double
handles.num_harmonics = max(1, floor(str2double(get(hObject, 'String'))));
init(hObject, handles);

% --- Executes on button press in previous_harmonic_pushbutton.
function previous_harmonic_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to previous_harmonic_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.current_harmonic > 1) handles.current_harmonic = handles.current_harmonic - 1;
end
guidata(hObject, handles);
set_current_harmonic(hObject, handles);

% --- Executes on button press in next_harmonic_pushbutton.
function next_harmonic_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to next_harmonic_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (handles.current_harmonic < handles.num_harmonics) handles.current_harmonic = handles.current_harmonic + 1;
end
guidata(hObject, handles);
set_current_harmonic(hObject, handles);

% --- Executes on button press in play_sound_pushbutton.
function play_sound_pushbutton_Callback(hObject, eventdata, handles)
% hObject    handle to play_sound_pushbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Fs = 44100;
num_seconds = 2;
fundamental_freq = 500; % 500Hz
t = 1:1/Fs:num_seconds;
x = zeros(1, length(t));
for n=1:handles.num_harmonics
    x = x + handles.amplitudes(n,1) * sin(2*pi*n*fundamental_freq*t + handles.phases(n,1)); 
end
sound(double(x)/(max(x) + 0.05),Fs);

% --------------------------------------------------------------------
function AboutMenuItem_Callback(hObject, eventdata, handles)
% hObject    handle to AboutMenuItem (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
msgbox('Sum of Sines - EE261 Autumn 2006 - Stanford University','About');


% --- CREATE FUNCTIONS ---

% --- Executes during object creation, after setting all properties.
function current_harmonic_popupmenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to current_harmonic_popupmenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes during object creation, after setting all properties.
function harmonics_listbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to harmonics_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function amplitude_slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to amplitude_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes during object creation, after setting all properties.
function phase_slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to phase_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes during object creation, after setting all properties.
function amplitude_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to amplitude_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function phase_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to phase_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function num_harmonics_edit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to num_harmonics_edit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function HelpMenu_Callback(hObject, eventdata, handles)
% hObject    handle to HelpMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

