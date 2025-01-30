class boundary_condition(object):
    def __init__(self):
        self.Dimension=None
        self.Discretization = None
        self.Origin=None
        self.Cell_centered=None
        self.Data=None

class pflotran_3D_gridded_dataset(object):
    ''' Gset_liquid_Pressure = pflotran_3D_gridded_dataset(xgrid, ygrid, zgrid)
        Gset_liquid_Pressure.get_west_boundary(liquid_Pressure_grid)
        Gset_liquid_Pressure.get_east_boundary(liquid_Pressure_grid)
        Gset_liquid_Pressure.get_south_boundary(liquid_Pressure_grid)
        Gset_liquid_Pressure.get_north_boundary(liquid_Pressure_grid)
        Gset_liquid_Pressure.get_bottom_boundary(liquid_Pressure_grid)
        Gset_liquid_Pressure.get_top_boundary(liquid_Pressure_grid)
        Gset_liquid_Pressure.get_initial_condition(liquid_Pressure_grid)'''
    def __init__(self, xgrid, ygrid, zgrid):
        #self.xgrid = xgrid 
        #self.ygrid = ygrid
        #self.zgrid = zgrid
        #self.dataset = value_grid
        self.xv = xgrid[0,:,0]
        self.yv = ygrid[:,0,0]
        self.zv = zgrid[0,0,:]
        (ny,nx,nz) = xgrid.shape
        self.nx = nx; self.ny = ny; self.nz = nz
        self.dx = self.xv[1] - self.xv[0]
        self.dy = self.yv[1] - self.yv[0]
        self.dz = self.zv[1] - self.zv[0]
        self.xmin = xv[0]-dx/2; self.xmax = xv[-1]+dx/2
        self.ymin = yv[0]-dy/2; self.ymax = yv[-1]+dy/2
        self.zmin = zv[0]-dz/2; self.zmax = zv[-1]+dz/2
        #print([xv[0]-dx/2, yv[0]-dy/2, zv[0]-dz/2])
        #print([xv[-1]+dx/2, yv[-1]+dy/2, zv[-1]+dz/2])

    def get_west_boundary(self, value_grid):
        west_boundary=boundary_condition()     
        x_ind = 0 
        west_boundary.Dimension = np.string_('YZ')
        west_boundary.Discretization = [self.dy, self.dz] 
        west_boundary.Origin = [self.yv[0]-self.dy/2, self.zv[0]-self.dz/2]
        west_boundary.Cell_centered = True
        west_boundary.Data = value_grid[:,x_ind,:]
        setattr(self,'west_boundary',west_boundary)

    def get_east_boundary(self, value_grid):
        east_boundary=boundary_condition()  
        x_ind = self.nx - 1 
        east_boundary.Dimension = np.string_('YZ')
        east_boundary.Discretization = [self.dy, self.dz] 
        east_boundary.Origin = [self.yv[0]-self.dy/2, self.zv[0]-self.dz/2]
        east_boundary.Cell_centered = True
        east_boundary.Data = value_grid[:,x_ind,:]
        setattr(self,'east_boundary',east_boundary)
    
    def get_south_boundary(self, value_grid):
        south_boundary=boundary_condition() 
        y_ind = 0
        south_boundary.Dimension = np.string_('XZ')
        south_boundary.Discretization = [self.dx, self.dz] 
        south_boundary.Origin = [self.xv[0]-self.dx/2, self.zv[0]-self.dz/2]
        south_boundary.Cell_centered = True
        south_boundary.Data = value_grid[y_ind,:,:]
        setattr(self,'south_boundary',south_boundary)

    def get_north_boundary(self,value_grid):
        north_boundary=boundary_condition() 
        y_ind = self.ny - 1 
        north_boundary.Dimension = np.string_('XZ')
        north_boundary.Discretization = [self.dx, self.dz] 
        north_boundary.Origin = [self.xv[0]-self.dx/2, self.zv[0]-self.dz/2]
        north_boundary.Cell_centered = True
        north_boundary.Data = value_grid[y_ind,:,:]
        setattr(self,'north_boundary',north_boundary)

    def get_bottom_boundary(self,value_grid):
        bottom_boundary=boundary_condition() 
        z_ind = 0
        bottom_boundary.Dimension = np.string_('XY')
        bottom_boundary.Discretization = [self.dx, self.dy] 
        bottom_boundary.Origin = [self.xv[0]-self.dx/2, self.yv[0]-self.dy/2]
        bottom_boundary.Cell_centered = True
        data = value_grid[:,:,z_ind]
        bottom_boundary.Data = data.transpose()
        setattr(self,'bottom_boundary',bottom_boundary)

    def get_top_boundary(self,value_grid):
        top_boundary=boundary_condition()
        z_ind = self.nz - 1
        top_boundary.Dimension = np.string_('XY')
        top_boundary.Discretization = [self.dx, self.dy] 
        top_boundary.Origin = [self.xv[0]-self.dx/2, self.yv[0]-self.dy/2]
        top_boundary.Cell_centered = True
        data = value_grid[:,:,z_ind]
        top_boundary.Data = data.transpose()
        setattr(self,'top_boundary',top_boundary)
    
    def get_initial_condition(self,value_grid):
        initial_condition = boundary_condition()
        initial_condition.Dimension = np.string_('XYZ')
        initial_condition.Discretization = [self.dx, self.dy, self.dz]
        initial_condition.Origin = [self.xv[0]-self.dx/2, self.yv[0]-self.dy/2, self.zv[0]-self.dz/2]
        initial_condition.Cell_centered = True
        initial_condition.Data = value_grid.transpose(1,0,2)
        setattr(self,'initial_condition',initial_condition)

    def add_new_attr(self, attr_name, value):
        #this is a testing function
        str = 'Warning! You just added {} in the object!'.format(attr_name)
        setattr(self,attr_name,value)
        print(str)
    
    def remove_attr(self, attr_name):
        #this is a testing function
        try:
            delattr(self,attr_name)
            str = 'Warning! You just deleted {} in the object!'.format(attr_name)
            print(str)
        except:
            print('There is no {} to removed from the object'.format(attr_name))

    def write_HDF5(self, condition, filename, group_name = None):
        '''f =  h5py.File('initial_Sg_STEP.h5','w')
                BC_grp = f.create_group('initial_gas_saturation') # create a material group
                BC_grp.create_dataset('Data', data=1 - Swgrid.transpose(1,0,2), dtype='f8')
                BC_grp.attrs['Dimension'] = np.string_('XYZ')
                BC_grp.attrs['Discretization'] = [dx,dy,dz]
                BC_grp.attrs['Origin'] = [xv[0]-dx/2, yv[0]-dy/2, zv[0]-dz/2]
                BC_grp.attrs['Cell Centered'] = True
                BC_grp.attrs['Interpolation Method'] = np.string_('STEP')
                #BC_grp.attrs['Interpolation Method'] = 'LINEAR'
                f.close()'''
        if group_name is None:
            A = filename.split('.')
            group_name = A[0]
        
        condition_=getattr(self, condition)
        f = h5py.File(filename,'w')
        fgrp = f.create_group(group_name) # create a material group
        fgrp.create_dataset('Data', data=condition_.Data, dtype='f8')
        fgrp.attrs['Dimension'] = condition_.Dimension
        fgrp.attrs['Discretization'] = condition_.Discretization
        fgrp.attrs['Origin'] = condition_.Origin
        fgrp.attrs['Cell Centered'] = condition_.Cell_centered
        f.close()