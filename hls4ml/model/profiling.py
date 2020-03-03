import importlib
from hls4ml.model.hls_model import HLSModel

libs = [('numpy', 'np'), ('pandas', 'pandas'), ('tensorflow', 'tensorflow'),
        ('seaborn', 'sb'), ('matplotlib.pyplot', 'plt')]
for (name, short) in libs:
    try:
        lib = importlib.import_module(name)
    except ImportError as error:
        print(error)
        print('Install hls4ml[profiling] extra depencies.')
    except Exception as exception:
        print(exception)
    else:
        globals()[short] = lib
globals()['keras'] = tensorflow.keras

def violinplot(data):
    f = plt.figure()
    hue = 'layer' if 'layer' in data.keys() else None
    vp = sb.violinplot(x='x', y='weight', hue=hue, data=data[data['x'] > 0])
    vp.set_yticklabels(vp.get_yticklabels(), rotation=45, ha='right')
    if hue is not None:
        vp.get_legend().remove()
    vp.set_xscale('log', basex=2)
    return f

def boxplot(data):
    from matplotlib.ticker import MaxNLocator
    f = plt.figure()
    hue = 'layer' if 'layer' in data.keys() else None
    vp = sb.boxplot(x='x', y='weight', hue=hue, data=data[data['x'] > 0], showfliers=False)
    vp.set_yticklabels(vp.get_yticklabels(), rotation=45, ha='right')
    if hue is not None:
        vp.get_legend().remove()
    vp.set_xscale('log', basex=2)
    return f

def histogram(data):
    from matplotlib.ticker import MaxNLocator
    # Power of 2 bins covering data range
    high = np.ceil(np.log2(max(data['x']))) + 1
    low = np.floor(np.log2(min(data[data['x'] > 0]['x']))) - 1
    bits = np.arange(low, high, 1)
    bins = 2 ** bits
    f = plt.figure()
    colors = sb.color_palette("husl", len(data['weight'].unique()))
    for i, weight in enumerate(data['weight'].unique()):
        x = data[data['weight'] == weight]['x']
        h, b = np.histogram(x, bins=bins)
        h = h * 1. / float(sum(h)) # normalize
        plt.bar(bits[:-1], h, width=1, fill=False, label=weight, edgecolor=colors[i])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('log2(x)')
    plt.ylabel('frequency')
    plt.legend()
    return f

def FacetGrid(data):
    hue = 'layer' if 'layer' in data.keys() else None
    vp = sb.FacetGrid(data[data['x'] > 0], row='weight', hue=hue)
    vp.map(sb.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    vp.map(sb.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
    vp.map(plt.axhline, y=0, lw=2, clip_on=False)
    vp.fig.subplots_adjust(hspace=-.25)
    return vp.fig

plots = {'violinplot' : violinplot,
         'boxplot' : boxplot,
         'FacetGrid' : FacetGrid,
         'histogram' : histogram}

def weights_hlsmodel(model):
    data = {'x' : [], 'layer' : [], 'weight' : []}
    for layer in model.get_layers():
        for iw, weight in enumerate(layer.get_weights()):
            w = weight.data.flatten()
            data['x'].extend(abs(w).tolist())
            data['layer'].extend([layer.name for i in range(len(w))])
            data['weight'].extend(['{}/{}'.format(layer.name, iw) for i in range(len(w))])

    data = pandas.DataFrame(data)
    return data

def weights_keras(model):
    data = {'x' : [], 'layer' : [], 'weight' : []}
    for layer in model.layers:
        name = layer.name
        weights = layer.get_weights()
        for i, w in enumerate(weights):
            w = w.flatten()
            n = len(w)
            data['x'].extend(abs(w).tolist())
            data['layer'].extend([name for j in range(n)])
            data['weight'].extend(['{}/{}'.format(name, i) for j in range(n)])

    data = pandas.DataFrame(data)
    return data

def types_boxplot(data):
    from matplotlib.patches import PathPatch
    from matplotlib.patches import Rectangle
    ax = plt.gca()
    f = plt.gcf()
    # Scale the data
    data['low'] = 2.**data['low']
    data['high'] = 2.**data['high']

    # Plot the custom precisions
    ticks = np.array([tick.get_text() for tick in plt.yticks()[1]])
    # Get the coordinates of the boxes to place the markers
    boxes = [c.get_extents().inverse_transformed(ax.transData) for c in ax.get_children() if isinstance(c, PathPatch)]
    ys = [(box.y0 + box.y1) / 2 for box in boxes]
    ys = [(y, y) for y in ys]
    for irow, row in data[data['layer'] != 'model'].iterrows():
        if row['layer'] in ticks:
            iy = np.argwhere(ticks == row['layer'])[0][0] # Determine which layer in the plot
            rectangle = Rectangle((row['low'], ys[iy][0]-0.4), row['high']-row['low'], 0.8, fill=True, color='grey', alpha=0.2)
            ax.add_patch(rectangle)

def types_histogram(data):
    ax = plt.gca()
    layers = np.array(ax.get_legend_handles_labels()[1])
    colors = sb.color_palette("husl", len(layers))
    ylim = ax.get_ylim()
    for irow, row in data[data['layer'] != 'model'].iterrows():
        if row['layer'] in layers:
            col = colors[np.argwhere(layers == row['layer'])[0][0]]
            plt.plot((row['low'], row['low']), ylim, '--', color=col)
            plt.plot((row['high'], row['high']), ylim, '--', color=col)

types_plots = {'boxplot' : types_boxplot,
               'histogram' : types_histogram}

def ap_fixed_WIF(type_str):
    if 'ap_fixed' in type_str:
        W = int(type_str.split(',')[0].split('<')[1])
        I = int(type_str.split(',')[1].split('>')[0])
        F = W - I
    elif 'ap_int' in type_str:
        W = int(type_str.replace('ap_int<','').replace('>',''))
        I = W
        F = 0
    else:
        W, I, F = 0, 0, 0
    return W, I, F

def types_hlsmodel(model):
    data = {'layer' : [], 'low' : [], 'high' : []}
    # Plot the default precision
    default_precision = model.config.model_precision['default']
    # assumes ap_fixed
    W, I, F = ap_fixed_WIF(default_precision)
    data['layer'].append('model')
    data['low'].append(-F)
    data['high'].append(I-1)

    for layer in model.get_layers():
        for iw, weight in enumerate(layer.get_weights()):
            wname = '{}/{}'.format(layer.name, iw)
            T = weight.type
            if T.name != 'model':
                W, I, F = ap_fixed_WIF(T.precision)
                data['layer'].append(wname)
                data['low'].append(-F)
                data['high'].append(I-1)
    data = pandas.DataFrame(data)
    return data

def activation_types_hlsmodel(model):
    data = {'layer' : [], 'low' : [], 'high' : []}
    # Get the default precision
    default_precision = model.config.model_precision['default']
    W, I, F = ap_fixed_WIF(default_precision)
    data['layer'].append('model')
    data['low'].append(-F)
    data['high'].append(I-1)
    for layer in model.get_layers():
        T = layer.get_output_variable().type.precision
        W, I, F = ap_fixed_WIF(T)
        data['layer'].append(layer.name)
        data['low'].append(-F)
        data['high'].append(I-1)
    data = pandas.DataFrame(data)
    return data

def activations_keras(model, X):
    # test layer by layer on data
    data = {'x' : [], 'weight' : []}
    partial_model = keras.models.Sequential()
    for layer in model.layers:
        print("   {}".format(layer.name))
        partial_model.add(layer)
        partial_model.compile(optimizer='adam', loss='mse')
        if not isinstance(layer, keras.layers.InputLayer):
            y = partial_model.predict(X).flatten()
            data['x'].extend(abs(y).tolist())
            data['weight'].extend([layer.name for i in range(len(y))])

    data = pandas.DataFrame(data)
    return data

def numerical(keras_model=None, hlsmodel=None, X=None, plot='boxplot'):
    """
    Perform numerical profiling of a model

    Parameters
    ----------
    model : keras model
        The keras model to profile
    X : array-like, optional
        Test data on which to evaluate the model to profile activations
        Must be formatted suitably for the model.predict(X) method
    plot : str, optional
        The type of plot to produce.
        Options are: 'boxplot' (default), 'violinplot', 'histogram', 'FacetGrid'

    Returns
    -------
    tuple
        The pair of produced figures. First weights and biases, then activations
    """

    print("Profiling weights")
    if hlsmodel is not None and isinstance(hlsmodel, HLSModel):
        data = weights_hlsmodel(hlsmodel)
    elif keras_model is not None and isinstance(keras_model, keras.Model):
        data = weights_keras(keras_model)
    else:
        print("Only keras and HLSModel models can currently be profiled")
        return False, False

    wp = plots[plot](data) # weight plot
    if isinstance(hlsmodel, HLSModel) and plot in types_plots:
        t_data = types_hlsmodel(hlsmodel)
        types_plots[plot](t_data)

    plt.title("Distribution of (non-zero) weights")
    plt.tight_layout()

    ap = None
    if X is not None and isinstance(keras_model, keras.Model):
        print("Profiling activations")
        data = activations_keras(keras_model, X)
        ap = plots[plot](data) # activation plot
        plt.title("Distribution of (non-zero) activations")
        plt.tight_layout()

    if X is not None and isinstance(hlsmodel, HLSModel):
        t_data = activation_types_hlsmodel(hlsmodel)
        types_plots[plot](t_data)

    return wp, ap

########COMPARE OUTPUT IMPLEMENTATION########
def get_ysim_from_file(project_dir, layer_names):
    """
    Get each layer's output from converted hls project. Note that the project
    has to be complied prior to using this method in Debug mode (i.e specify Debug: True in config file).
    
    You have to run get_ymodel_keras(keras_model, X) first to obtain ymodel, then use the keys of
    ymodel to pass to layer_names. This is because we want identical keys betwen ymodel and ysim.
    
    Example of how to use this:

    ymodel = get_ymodel_keras(keras_model, X)
    ysim = get_ysim_from_file(project_dir, list(ymodel.keys()))

    Params:
    ------
    project_dir : string
        Relative path to the hls project's directory
    layer_names : list
        A list of layer's names in the model. (Obtained via ymodel.keys()) 

    Return:
    ------
        A dictionary in the form {"layer_name": ouput array of layer in hls model}
    """
    ysim = {}

    for layer in layer_names:
        print("Processing {} in HLS model...".format(layer))
        ysim[layer] = np.loadtxt('{}/tb_data/{}_output.log'.format(project_dir, layer)).flatten()

    print("Done taking outputs for HLS model.")
    return ysim

def _is_ignored_layer(layer):
    """Some layers need to be ingored during inference"""
    if isinstance(layer, (keras.layers.InputLayer,
                        keras.layers.Dropout, 
                        keras.layers.Flatten)):
        return True
    return False

def _add_layer_get_ouput(partial_model, layer, X):
    copy_model = keras.models.clone_model(partial_model) #Make a copy to avoid modify the original model
    copy_model.add(layer)
    copy_model.compile(optimizer='adam', loss='mse')

    y = copy_model.predict(X).flatten()

    return y

def get_ymodel_keras(keras_model, X):
    """
    Calculate each layer's ouput and put them into a dictionary

    Params:
    ------
    keras_model: a keras model
    X : array-like
        Test data on which to evaluate the model to profile activations
        Must be formatted suitably for the model.predict(X) method

    Return:
    ------
        A dictionary in the form {"layer_name": ouput array of layer}
    """
    
    partial_model = keras.models.Sequential()
    ymodel = {}
    
    for layer in keras_model.layers:
        print("Processing {} in Keras model...".format(layer.name))
        if not _is_ignored_layer(layer):
            #If the layer has activation integrated then separate them
            #Note that if the layer is a standalone activation layer then skip this
            if hasattr(layer, 'activation') and not isinstance(layer,keras.layers.Activation):
                if layer.activation:
                    
                    if layer.activation.__name__ == "linear":
                        ymodel[layer.name] = _add_layer_get_ouput(partial_model, layer, X)
                    
                    else:
                        temp_activation = layer.activation
                        layer.activation = None

                        #Get output for layer without activation
                        ymodel[layer.name] = _add_layer_get_ouput(partial_model, layer, X)

                        #Get ouput for activation
                        ymodel[layer.name + "_{}".format(temp_activation.__name__)] = temp_activation(ymodel[layer.name])
                        
                        #Add the activation back
                        layer.activation = temp_activation
            else:    
                ymodel[layer.name] = _add_layer_get_ouput(partial_model, layer, X)
        
        #Add the layer for later processing
        partial_model.add(layer)

    print("Done taking outputs for Keras model.")
    return ymodel

def _norm_diff(ymodel, ysim):
    """Calculate the square root of the sum of the squares of the differences"""
    diff = {}
    
    for key in list(ymodel.keys()):
        diff[key] = np.linalg.norm(ysim[key]-ymodel[key])
    
    #---Bar Plot---
    f, ax = plt.subplots()

    plt.bar(list(diff.keys()),list(diff.values()))
    plt.title("layer-by-layer output differences")
    ax.set_ylabel('Norm of difference vector')
    plt.xticks(rotation=90)
    plt.tight_layout()

    return f

def _dist_diff(ymodel, ysim):
    """
    Calculate the normalized distribution of the differences of the elements
    of the output vectors. 
    If difference >= original value then the normalized difference will be set to 1,
    meaning "very difference".
    If difference < original value then the normalized difference would be difference/original.
    """

    diff = {}

    for key in list(ymodel.keys()):
        diff_vector = np.absolute(ymodel[key] - ysim[key])
        abs_ymodel = np.absolute(ymodel[key])
    
        normalized_diff = np.zeros(diff_vector.shape)
        normalized_diff[diff_vector >= abs_ymodel] = 1
        
        #Fill out the rest
        index = diff_vector < abs_ymodel
        normalized_diff[index] = diff_vector[index] / abs_ymodel[index]
        
        diff[key] = normalized_diff
    
    #---Box Plot---
    f, ax = plt.subplots()
    pos = np.array(range(len(list(diff.values())))) + 1            
    ax.boxplot(list(diff.values()), sym='k+', positions=pos)
    
    #--formatting
    plt.title("Layer-by-layer distribution of output differences")
    ax.set_xticklabels(list(diff.keys()))
    ax.set_ylabel('Percent difference.')
    plt.xticks(rotation=90)
    plt.tight_layout()

    return f


def compare(keras_model, hls_model, X, file_based=True, plot_type = "dist_diff"):
    """
    Compare each layer's output in keras and hls model

    Params:
    ------
    keras_model : original keras model
    hls_model : converted HLS model
    file_based : (boolean) whether the comparison is based on csim output files, 
                 or memory based.
    plot_type : (string) different methods to visualize the y_model and y_sim differences.
                Possible options include:
                     - "norm_diff" : square root of the sum of the squares of the differences 
                                    between each output vectors 
                     - "dist_diff" : The normalized distribution of the differences of the elements
                                    between two output vectors
        

    Return:
    ------
        plot object of the histogram depicting the difference in each layer's ouput
    """
    
    #Take in output from both models
    #Note that each y is a dictionary with structure {"layer_name": flattened ouput array}
    ymodel = get_ymodel_keras(keras_model, X)

    if file_based:
        ysim = get_ysim_from_file(hls_model.config.get_output_dir(), list(ymodel.keys()))
    #else: to be implemented 
    
    print("Plotting difference...")
    f = plt.figure()

    if plot_type == "norm_diff":
        f = _norm_diff(ymodel, ysim)

    elif plot_type == "dist_diff":
        f = _dist_diff(ymodel, ysim)

    return f