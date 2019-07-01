import sys
sys.path.append('..')
import unittest
import yaml
import os

class ConfigFileTestCase(unittest.TestCase):
    """
    This test suite runs a series of tests to ensure that the config
    file has all the necessary key-value pairs and that the values
    obey the necessary constraints
    """

    def setUp(self):
        CWD_PATH = os.getcwd()
        config_path = os.path.join(CWD_PATH,'configs/config.yml')
        with open(config_path, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
        self.cfg = cfg

    def test_training_parameters(self):
        """"
        Test to make sure the default parameters for learning and rolling out are valid
        """

        self.assertIsInstance(self.cfg['num_learning_steps'],int,
            'Number of learning steps must be an integer')
        self.assertTrue(self.cfg['num_learning_steps']>0,'Number of learning steps must be greater than zero')
        self.assertIsInstance(self.cfg['num_rollout_steps'],int,
            'Number of rollout steps must be an integer')
        self.assertTrue(self.cfg['num_rollout_steps']>0,'Number of learning steps must be greater than zero')
        self.assertTrue(self.cfg['learning_rate_val']>0,'Learning rate must be > 0')
    def test_environment_parameters(self):
        """
        Tests to make sure the parameters of the vibrating bridge environment are valid
        """
        # First test for positivity
        positive_param_list = ['time_interval','wave_speed','system_length','num_lattice_points',
        'drive_magnitude','num_warmup_steps','num_equi_steps','timepoints_per_step','max_steps',
        'num_force_points','force_width','max_u','max_force']
        for param in positive_param_list:
            error_string = '{} must be > 0'.format(param)
            self.assertTrue(self.cfg[param]>0,error_string)
        # Test for params that must be < 0
        negative_param_list = ['min_u','min_force']
        for param in negative_param_list:
            error_string = '{} must be < 0'.format(param)
            self.assertTrue(self.cfg[param]<0,error_string)
        # Test for params that must be integers
        int_param_list = ['num_lattice_points','num_warmup_steps','num_equi_steps',
        'timepoints_per_step','max_steps','num_force_points']
        for param in int_param_list:
            error_string = '{} must be an integer'.format(param)
            self.assertIsInstance(self.cfg[param],int,error_string)


        """
        self.assertTrue(self.cfg['time_interval']>0,'time interval must be > 0')
        self.assertTrue(self.cfg['wave_speed']>0,'wave speed must be > 0')
        self.assertTrue(self.cfg['system_length']>0,'system length must be > 0')
        self.assertIsInstance(self.cfg['num_lattice_points'],int,'must have integer number of lattice points')
        self.assertTrue(self.cfg['num_lattice_points']>0,'number of lattice points must be > 0')
        self.assertTrue(self.cfg['drive_magnitude']>0,'drive magnitude must be > 0')
        self.assertIsInstance(self.cfg['num_warmup_steps'],int,'must have integer number of warmup steps')
        self.assertTrue(self.cfg['num_warmup_steps']>0,'number of warmup steps must be > 0')
        self.assertIsInstance(self.cfg['num_equi_steps'],int,'must have integer number of equilibriation steps')
        self.assertTrue(self.cfg['num_equi_steps']>0,'number of equilibriation steps must be > 0')
        self.assertIsInstance(self.cfg['timepoints_per_step'],int,'must have integer number of timepoints per step')
        self.assertTrue(self.cfg['timepoints_per_step']>0,'number of timepoints per step must be > 0')
        """
if __name__ == '__main__':
    unittest.main()
        