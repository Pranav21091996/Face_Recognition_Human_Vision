def sample_z(m, n):
    return np.random.normal(loc=0.0, scale=1.0, size=(m, n)).astype(np.float32)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = y.get_shape()
    y = tf.reshape(y, (x_shapes[0], 1, 1, y_shapes[1]))
    z = tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[1]])
    y = tf.cast(y,'float32')
    y = y*z

    return tf.concat([x , y], 3)


def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):
    """
    Spatial Transformer Network layer implementation as described in [1].
    The layer is composed of 3 elements:
    - localisation_net: takes the original image as input and outputs 
      the parameters of the affine transformation that should be applied
      to the input image.
    - affine_grid_generator: generates a grid of (x,y) coordinates that 
      correspond to a set of points where the input should be sampled 
      to produce the transformed output.
    - bilinear_sampler: takes as input the original image and the grid
      and produces the output image using bilinear interpolation.
    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
      transformer layer is at the beginning of architecture. Should be 
      a tensor of shape (B, H, W, C). 
    - theta: affine transform tensor of shape (B, 6). Permits cropping, 
      translation and isotropic scaling. Initialize to identity matrix. 
      It is the output of the localization network.
    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)
    """
    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]
    C = tf.shape(input_fmap)[3]

    # reshape theta to (B, 2, 3)
    theta = tf.reshape(theta, [B, 2, 3])

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

    return out_fmap

def tf_compute_transformation_matrix(bbox, shape=16., img_height=64):
    rel_factor = float(shape)/img_height
    x, y, w, h = rel_factor*bbox[0], rel_factor*bbox[1], rel_factor*bbox[2], rel_factor*bbox[3]

    t_x = (x+0.5*w-0.5*shape)/(0.5*shape)
    t_y = (y+0.5*h-0.5*shape)/(0.5*shape)

    scale_x = (w / shape)
    scale_y = (h / shape)

    line0 = tf.stack((scale_x, 0.0, t_x))
    line1 = tf.stack((0.0, scale_y, t_y))
    transformation_matrix = tf.concat((line0, line1), axis=0)
    transformation_matrix = tf.reshape(transformation_matrix, (2, 3))

    return transformation_matrix 
  


def tf_compute_transformation_matrix_inverse(bbox, shape=16., img_height=64):
    rel_factor = float(shape)/img_height
    x, y, w, h = rel_factor*bbox[0], rel_factor*bbox[1], rel_factor*bbox[2], rel_factor*bbox[3]

    scale_x = (shape / w)
    scale_y = (shape / h)

    t_x = (shape - 2 * x) / w - 1
    t_y = (shape - 2 * y) / h - 1

    line0 = tf.stack((scale_x, 0.0, t_x))
    line1 = tf.stack((0.0, scale_y, t_y))

    transformation_matrix = tf.concat((line0, line1), axis=0)
    transformation_matrix = tf.reshape(transformation_matrix, (2, 3))

    return transformation_matrix  
  
def sample_gen_label(mb_size):
    labels = np.random.multinomial(1, 200*[1/200], size=mb_size)
    return labels.astype(np.float32)


def sample_gen_label_sorted(mb_size, label_dim=10):
    labels = np.zeros((mb_size, label_dim))
    for idx in range(label_dim):
        labels[idx*label_dim:idx*label_dim+label_dim, idx] = 1
    return labels.astype(np.float32)


def sample_bbox(mb_size):
    pos_box_x = np.random.randint(low=0, high=44, size=(mb_size, 1))
    pos_box_y = np.random.randint(low=0, high=44, size=(mb_size, 1))

    coin = np.random.binomial(1, 0.5)
    if coin < 0.1:
        scale_box_x = np.random.randint(low=8, high=16, size=(mb_size, 1))
    elif 0.1 < coin < 0.8:
        scale_box_x = np.random.randint(low=12, high=18, size=(mb_size, 1))
    else:
        scale_box_x = np.random.randint(low=16, high=21, size=(mb_size, 1))
    scale_box_y = np.random.randint(low=18, high=21, size=(mb_size, 1))

    boxes = np.concatenate((pos_box_x, pos_box_y, scale_box_x, scale_box_y), axis=1)
    return boxes.astype(np.float32)


def sample_bbox_sorted(mb_size, sort_size=False):
    pos_box_x = np.zeros((mb_size, 1))
    for idx in range(100):
        pos_box_x[idx, 0] = (idx % 10 + 1) * 4

    pos_box_y = np.zeros((mb_size, 1))
    for idx in range(10):
        pos_box_y[idx * 10:idx * 10 + 10, 0] = (idx + 1) * 4

    if sort_size:
        scale_box_x = np.zeros((mb_size, 1))
        for idx in range(100):
            scale_box_x[idx, 0] = (idx % 10 + 1) + 10

        scale_box_y = np.zeros((mb_size, 1))
        for idx in range(10):
            scale_box_y[idx * 10:idx * 10 + 10, 0] = (idx + 1) + 10
    else:
        scale_box_x = np.random.randint(low=16, high=21, size=(mb_size, 1))
        scale_box_y = np.random.randint(low=18, high=21, size=(mb_size, 1))

    boxes = np.concatenate((pos_box_x, pos_box_y, scale_box_x, scale_box_y), axis=1)
    return boxes.astype(np.float32)


def sample_generator_input(mb_size, n, sort_labels=False, sort_location=False, sort_bbox_size=False):
    _z = sample_z(mb_size, n)
    _Y = sample_gen_label_sorted(mb_size) if sort_labels else sample_gen_label(mb_size)

    if sort_bbox_size:
        _bbox = sample_bbox_sorted(mb_size, sort_size=True)
    else:
        _bbox = sample_bbox_sorted(mb_size) if sort_location else sample_bbox(mb_size)


    return _z, _Y, _bbox

def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature 
    map, will create an output feature map that is an 
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample. 
    - width: desired width of grid/output. Used
      to downsample or upsample. 
    - theta: affine transform matrices of shape (num_batch, 2, 3). 
      For each image in the batch, we have 6 theta parameters of 
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized gird (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the 
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation, 
         and isotropic scaling.
    """
    # grab batch size
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the 
    normalized coordinates provided by the sampling grid. Note that 
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - interpolated images according to grids. Same size as grid.
    """
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # rescale x and y to [0, W/H]
    x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])


    return out  
  
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)  

def transform(img):
    img = np.divide(np.add(img,1),2.0)
    return img
  
def process_oneimg(img):
    img = np.subtract(np.multiply(img , 2),1.0)
    return img

def lrelu(x,alpha=0.2):
	return tf.maximum(x,alpha*x)	

def conv2d(x, features, kernel=[4,4], strides=[1,2,2,1], name="conv_layer"):
	with tf.variable_scope(name) as scope:
		weights = weight(shape=kernel + features, name="weights")
		biases = bias(shape=[features[-1]], name="bias")
		output = tf.nn.conv2d(x, weights, strides=strides, padding='SAME') 
		output = tf.nn.bias_add(output, biases)
		return output	

def deconv2d(x, features, output_shape, kernel=[4,4], strides=[1,2,2,1], name="deconv_layer"):
	with tf.variable_scope(name) as scope:
		weights = weight(shape=kernel + features, name="weights")
		biases = bias(shape=[features[0]], name="bias")
		output = tf.nn.conv2d_transpose(x, weights, output_shape=output_shape, strides=strides, padding='SAME') 
		return tf.reshape(tf.nn.bias_add(output, biases), output.get_shape())

def bias(shape, name):
	return tf.get_variable(name, shape,initializer=tf.constant_initializer(0.00000))

def weight(shape, name):
	return tf.get_variable(name, shape,initializer=tf.glorot_uniform_initializer())	

def dense(x, shape, name):
	with tf.variable_scope(name):
		weights = weight(shape, name="weights")
		biases = bias([shape[-1]], name="bias")
		return tf.matmul(x,weights) + biases

def batch_norm(inputs, decay=0.9, epsilon=0.00001, scale=True, isTrain=True, name="batch_norm"):
	return tf.contrib.layers.batch_norm(inputs, decay=decay, scale=scale, epsilon=epsilon, updates_collections=None, is_training=isTrain, scope=name)			

