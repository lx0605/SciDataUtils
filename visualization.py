
import matplotlib.pyplot as plt, plotly.io as pio, plotly.graph_objects as go, plotly.express as px

def generate_colors(n_colors, cmap=None):
    ''' automatically generate discrete color scales from a continuous color map
    '''
    if cmap is None:
        cmap = 'turbo'

    colors = px.colors.sample_colorscale(cmap, [n/(n_colors -1) for n in range(n_colors)])
    c_array = []
    
    for i in range(0, n_colors):     
        c_str = colors[i]
        temp_array_1 = [i/n_colors, c_str]
        c_array.append(temp_array_1)
        temp_array_2 = [(i+1)/n_colors, c_str]
        c_array.append(temp_array_2)

    return colors, c_array


def plotly_multi(data):
    '''making plots with multiple y-axis
    index: x data
    column 1-2: y data
    # https://stackoverflow.com/questions/65037641/plotly-how-to-add-multiple-y-axes'''
    if data.shape[1]>2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data.index, y=data.iloc[:, 0], name=data.columns[0]))
    
        fig.update_layout(
            xaxis=dict(domain=[0.1, 0.9]),
            yaxis=dict(title=data.columns[0]),
            yaxis2=dict(title=data.columns[1], anchor="x", overlaying="y", side="right"))
    
        for i, col in enumerate(data.columns[1:], 1):
            fig.add_trace(
                go.Scatter(x=data.index,y=data[col],name=col,yaxis=f"y{i+1}"))
    
        for i, col in enumerate(data.columns[2:], 2):
            axis = f"yaxis{i+1}"
    
            if i%2 == 0:
                side = "left"
                position = (i-1)*0.05
            else:
                side = "right"
                position = 1 - (i-2)*0.05
    
            axis_value = dict(
                title=col,
                anchor="free",
                overlaying="y",
                side=side,
                position=position)
            exec(f"fig.update_layout({axis} = axis_value)")
    if data.shape[1]==2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
            go.Scatter(x=data.index, y=data.iloc[:, 0], name=data.columns[0]),
            secondary_y=False,)
        fig.add_trace(
            go.Scatter(x=data.index, y=data.iloc[:, 1], name=data.columns[1]),
            secondary_y=True,)
        # Set x-axis title
        fig.update_xaxes(title_text="Date")
        # Set y-axes titles
        fig.update_yaxes(title_text=data.columns[0], secondary_y=False)
        fig.update_yaxes(title_text=data.columns[0], secondary_y=True)
    if data.shape[1] == 1:
        fig = px.line(data.reset_index(), x = data.index.name, y = data.columns)
    
    fig.update_layout(
    title_text="Data",
    width=800,)
    
    fig.show()
   
def collect_edges(tri):
    '''get edges from triangles'''
    edges = set()

    def sorted_tuple(a,b):
        return (a,b) if a < b else (b,a)
    # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
    for (i0, i1, i2, i3) in tri.simplices:
        edges.add(sorted_tuple(i0,i1))
        edges.add(sorted_tuple(i0,i2))
        edges.add(sorted_tuple(i0,i3))
        edges.add(sorted_tuple(i1,i2))
        edges.add(sorted_tuple(i1,i3))
        edges.add(sorted_tuple(i2,i3))
    return edges

def plotly_tri(tri):
    '''plot triangulation'''
    points = tri.points
    #poly 3D trianglarte delaunay using plotly with edge and scattered points 
    edges = collect_edges(tri)
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for (i,j) in edges:
        x = np.append(x, [points[i, 0], points[j, 0], np.nan])      
        y = np.append(y, [points[i, 1], points[j, 1], np.nan])      
        z = np.append(z, [points[i, 2], points[j, 2], np.nan])
    lines = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='green',width =1.5))

    scatter = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(color='blue',size=4))
    pfig = go.Figure(data=[scatter, lines])
    # Set the layout
    pfig.update_layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(eye=dict(x=1.25, y=1.25, z=1.25))
        )
    )
    pfig.show()

def plotly_z_slice_contour(X, Y, Z, zmin=None, zmax=None, xaxis_dict=None, yaxis_dict=None, fig_title_str=None, cbar_title_str=None, margin_dict=None):
    '''plot 3D contour plot with z slices in plotly'''
    if margin_dict is None:
        margin_dict = {'l': 30, 'r': 0, 'b': 40, 't': 40, 'pad': 0}
    if xaxis_dict is None:
        xaxis_dict = {'title': 'default x', 'title_standoff': 0}
    if yaxis_dict is None:
        yaxis_dict = {'title': 'default y', 'title_standoff': 0}
    if fig_title_str is None:
        #title_str = 'xy plane field @ {} '.format(z_grid[0,0,n]*-1),
        fig_title_str = 'xy plane field'
    if cbar_title_str is None:
        cbar_title_str = ' '
    if zmin is None:
        zmin = np.min(Z)
    if zmax is None:
        zmax = np.max(Z)

    label_font_dict = {'size':12, 'color':'white'}
    colorbar_font_dict = {'size': 14, 'family':'Arial, sans-serif'}
    contour_dict = {'coloring' :'heatmap', 'showlabels' : True, 'labelfont': label_font_dict}
    colorbar_dict = {'title': cbar_title_str, 'titleside':'right','titlefont':colorbar_font_dict}
    color_dict = {'zmin': zmin, 'zmax': zmax, 'colorscale': 'viridis', 'colorbar': colorbar_dict}
    fig = go.Figure(data =
        go.Contour(x = X, y = Y, z = Z,
               #x = x_grid[0,:,0], y = y_grid[:,0,0], z = vel_grid[:,:,n],
                **color_dict,
               contours = contour_dict,
               ), #go.Contour
               layout=go.Layout(height=600, width=800,
               title = fig_title_str,
               xaxis=xaxis_dict,
               yaxis=yaxis_dict,
               margin=margin_dict
               )#go.Layout
               )#go.Figure
    #fig.update_layout(margin=dict(l=50,r=50,b=50,t=50,pad=0))
    #fig.update_xaxes(title_text = 'x (m)', title_standoff=0)
    #fig.update_yaxes(title_text = 'y (m)', title_standoff=0)
    return fig

def plotly_multi_slice_contour(x_grid,y_grid,z_grid,v_grid,ns=None,nt=5,cmin=None,cmax=None,cb_title=None,
    fig_title='default fig title',
    scene_dict={'xaxis_title':'X AXIS TITLE', 'yaxis_title':'Y AXIS TITLE','zaxis_title':'Z AXIS TITLE'},
    layout_dict={'height': 800, 'width': 600,'margin': None,'title':None},
    margin = {'l': 30, 'r': 0, 'b': 40, 't': 40, 'pad': 0}):
    ## end of input line

    layout_dict['margin'] = margin
    layout_dict['title'] = fig_title
    #layout_dict = dict(layout_dict,margin)
    ny, nx, nz = v_grid.shape
    #print(nz)
    x2d = x_grid[0,:,0]
    y2d = y_grid[:,0,0]
    z_lvs = z_grid[0,0,:]
    data = []
    if ns is None:
        ns = np.linspace(0, nz-1, nt, dtype=int)
        print('z_indices:')
        print(ns)
    if cmin is None:
        cmin = np.min(v_grid)
    if cmax is None:
        cmax = np.max(v_grid)
    if cb_title is None:
        cb_title = 'default cb_title'
    showscale = np.zeros(ns.shape, dtype=bool)
    showscale[0] = 1

    colorbar_font_dict = {'size': 14, 'family':'Arial, sans-serif'}
    colorbar_dict = {'title': cb_title, 'titleside':'right','titlefont':colorbar_font_dict}
    color_dict = {'cmin': cmin, 'cmax': cmax, 'colorscale': 'viridis', 'colorbar': colorbar_dict}
        
    for k in range(0, len(ns)):
        i = ns[k]
        z_ele = np.ones(v_grid[:,:,0].shape) * z_lvs[i] # this is the elevation of the surface, which can be obtained from the z grids
        #print(showscale[k])
        v_dat = v_grid[:,:,i]
        plot_dict = {'x': x2d, 'y': y2d, 'z': z_ele, 'surfacecolor': v_dat}
        data.append(go.Surface({**plot_dict, **color_dict, 'showscale': bool(showscale[k])}))
    
    #layout_dict = {'height':800, 'width': 600, 'title': fig_title, 'xaxis':x_axis,'yaxis':y_axis,'margin':margin}
    #layout_dict = {'height':800, 'width': 600}
    fig = go.Figure(data)
    #fig.update_xaxes(x_axis)
    #fig.update_yaxes(y_axis)
    fig.update_layout(scene=scene_dict,**layout_dict)

    return fig

def plotly_3D_volume(x_grid,y_grid,z_grid,v_grid,nt_z=5,y_loc=None, x_loc=None, cmin=None,cmax=None,cb_title=None,isomax=None, isomin=None,
    zslice_dict = None, yslice_dict = None, xslice_dict = None, opacity=0.8,surface_count=None,
    fig_title='default fig title',
    surface_dict={'fill': 0.5, 'pattern': 'odd'},
    caps_dict={'x_show': False, 'y_show': False, 'z_show': False},
    scene_dict={'xaxis_title':'X AXIS TITLE', 'yaxis_title':'Y AXIS TITLE','zaxis_title':'Z AXIS TITLE'},
    layout_dict={'height': 800, 'width': 600,'margin': None,'title':None},
    margin = {'l': 30, 'r': 0, 'b': 40, 't': 40, 'pad': 0}, colorbar_dict=None, color_scale=None,
    fig_fn=None):
    
    layout_dict['margin'] = margin
    layout_dict['title'] = fig_title

    # obtain the array shape
    ny, nx, nz = v_grid.shape

    xf = x_grid.flatten()
    yf = y_grid.flatten()
    zf = z_grid.flatten()
    vf = v_grid.flatten()

    if zslice_dict is None:
        ns = np.linspace(0, nz-1, nt_z, dtype=int)
        z_e = z_grid[0,0,:]
        z_loc = z_e[ns]
        zslice_dict = {'show': True, 'locations': z_loc}
    
    if yslice_dict is None:
        if y_loc is None:
            y_e = y_grid[:,0,0]
            y_loc = np.asarray((np.max(y_e) + np.min(y_e))/2)
        yslice_dict = {'show': True, 'locations': y_loc}
#         ns = np.linspace(0, ny-1, nt_y, dtype=int)
#         y_e = y_grid[:,0,0]
#         y_loc = y_e[ns]
#         yslice_dict = {'show': True, 'locations': y_loc}
    
    if xslice_dict is None:
        if x_loc is None:
            x_e = x_grid[0,:,0]
            x_loc = np.asarray((np.max(x_e) + np.min(x_e))/2)
        xslice_dict = {'show': True, 'locations': x_loc}
#         ns = np.linspace(0, nx-1, nt_x, dtype=int)
#         x_e = x_grid[0,:,0]
#         x_loc = x_e[ns]
#         xslice_dict = {'show': True, 'locations': x_loc}

    if cmin is None:
        cmin = np.min(v_grid)
    if cmax is None:
        cmax = np.max(v_grid)
    
    var_dict = {'x': xf, 'y': yf, 'z': zf, 'value': vf}

    colorbar_font_dict = {'size': 14, 'family':'Arial, sans-serif'}

    if colorbar_dict is None:
        colorbar_dict = {'title': cb_title, 'titleside':'right','titlefont':colorbar_font_dict}
    
    if color_scale is None:
        color_scale = 'viridis'
    
   
    color_dict = {'cmin': cmin, 'cmax': cmax, 'colorscale': color_scale, 'colorbar': colorbar_dict}

    
    fig = go.Figure(data=go.Volume(
    #x=xf,y=yf,z=zf,value=vf,
    **var_dict,
    isomin=isomin,
    isomax=isomax,
    opacity=opacity,
    surface_count=surface_count,
    slices_z=zslice_dict, slices_y=yslice_dict,slices_x=xslice_dict,
    surface=surface_dict, **color_dict,
    caps= caps_dict # caps on planes
    ))

    fig.update_layout(scene=scene_dict,**layout_dict)

    if fig_fn is not None:
        #fig_fn = 'plot_of_{}_'.format(np.max(df_in[c].values))
        fig.write_html(fig_fn+'.html')
    
    return fig

def marginal_scatter(df_in, x=None, y=None, c=None, range_color = None, x_margin='histogram',y_margin='histogram',
    width= 800, height=800,
    xaxis={'nticks': 5, 'range': None}, yaxis={'nticks': 5, 'range': None}, 
    margin={'l': 30, 'r': 0, 'b': 40, 't': 40, 'pad': 0}, fig_fn = None, color_sequence=None):


    col_names = df_in.columns
    
    
    #color_sequence = plotly.colors.sequential.Viridis
    if color_sequence is None:
        color_sequence = plotly.colors.sequential.Viridis
  
    if x is None:
        x = col_names[0]
    if y is None:
        y = col_names[1]
    if c is None:
        c = col_names[2]
    if range_color is None:
        rmin = np.min(df_in[c].values)
        rmax = np.max(df_in[c].values)
        range_color = [rmin, rmax]
        #print(range_color)


#     fig = px.scatter(df_in, x=x, y=y, color=c, marginal_x=x_margin, marginal_y=y_margin, 
#                     color_continuous_scale = color_sequence, range_color = range_color)
    fig = px.scatter(df_in, x=x, y=y, color=c, marginal_x=x_margin, marginal_y=y_margin, 
                     color_continuous_scale=  color_sequence, range_color = range_color)

    fig.update_layout(scene = dict(
        xaxis = xaxis,
        yaxis = yaxis,
        ),
    width=width,
    height=height,
    margin=margin)

    if fig_fn is not None:
        #fig_fn = 'plot_of_{}_'.format(np.max(df_in[c].values))
        fig.write_html(fig_fn+'.html')
                
    return fig

def make_profile_plot(y,df,fig_prop):
    lxc=['b','g','r','c','m','k']
    lxls=['-','--','-.',':']
    lxmarker=['.','s','x','^','d','P']
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
      
    for n in np.arange(len(df.columns)):
        n_color=np.remainder(n,len(lxc)).astype(int)
        n_ls=np.remainder(n,len(lxls)).astype(int)
        n_marker=np.remainder(n,len(lxmarker)).astype(int)
        ax.plot(y,df.iloc[:,n],c=lxc[n_color],ls=lxls[n_ls],marker=lxmarker[n_marker],label=df.columns[n],markersize=fig_prop.markersize)
    
    
    
    ax.set_ylim(fig_prop.ylim)
    ax.set_xlim(fig_prop.xlim)
    ax.set_xscale(fig_prop.xscale)
    ax.set_yscale(fig_prop.yscale)

    ax.set_xlabel(fig_prop.xlabel)
    ax.set_ylabel(fig_prop.ylabel)
    
    plt.legend(loc=fig_prop.legend_loc)
    plt.title(fig_prop.title)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=0.8, top=1, wspace=0.2, hspace=0.2)