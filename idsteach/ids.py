# IDSTeach: Generate data to teach continuous categorical data.
# Copyright (C) 2015  Baxter S. Eaves Jr.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from idsteach import utils
from scipy.misc import comb as nchoosek

import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3
import seaborn as sns

import copy

sns.set_palette("gray")

# FIXME: get relpath
DEFAULT_HILLENBRAND = '../data/hillenbrand.csv'
# _________________________________________________________________________________________________
# Data taken from hillenbrand (F1 and F2)
# `````````````````````````````````````````````````````````````````````````````````````````````````
all_the_vowels = ['AE', 'AH', 'AW', 'EH', 'EI', 'ER', 'IH', 'IY', 'OA', 'OO', 'UH', 'UW']
corner_vowels = ['AH', 'IY', 'UW']  # FIXME: check this

hillenbrand_data = {
    'AE': {
        'data': np.array([
            [678, 2293], [624, 2442], [666, 2370], [743, 2230], [677, 2320], [627, 2266],
            [690, 2327], [658, 2650], [685, 2299], [621, 2249], [868, 2004], [682, 2486],
            [668, 2252], [726, 2350], [620, 2316], [634, 2596], [672, 2145], [698, 2339],
            [586, 2299], [616, 2156], [576, 2429], [674, 2256], [738, 2378], [646, 2406],
            [746, 1944], [734, 2518], [662, 2276], [696, 2447], [687, 2378], [645, 2154],
            [626, 2374], [689, 2701], [557, 2586], [668, 2296], [746, 2371], [893, 2070],
            [665, 2408], [564, 2442], [714, 2254], [625, 2594], [552, 2227], [685, 2205],
            [657, 2192], [649, 2508], [817, 2102], [626, 2331], [706, 2400], [751, 2432]],
            dtype=np.dtype(float)),
        'example': 'pat',
        'unicode': 'æ',
    },
    'AH': {
        'data': np.array([
            [1012, 1603], [883, 1682], [1025, 1548], [804, 1484], [935, 1377], [804, 1363],
            [939, 1233], [827, 1701], [913, 1436], [856, 1540], [882, 1380], [887, 1743],
            [869, 1495], [994, 1609], [955, 1615], [1085, 1687], [937, 1623], [918, 1640],
            [818, 1351], [769, 1451], [1008, 1495], [905, 1514], [1063, 1680], [1053, 1677],
            [798, 1351], [869, 1751], [863, 1538], [938, 1462], [957, 1591], [827, 1543],
            [708, 1547], [1163, 1685], [1011, 1541], [1035, 1486], [952, 1676], [810, 1314],
            [993, 1671], [931, 1348], [822, 1371], [875, 1491], [884, 1568], [938, 1492],
            [914, 1383], [901, 1509], [975, 1534], [796, 1502], [968, 1433]],
            dtype=np.dtype(float)),
        'example': 'hod',
        'unicode': 'ɑ',
    },
    'AW': {
        'data': np.array([
            [815, 1200], [760, 1225], [772, 1051], [770, 1245], [923, 1368], [778, 1220],
            [809, 1097], [815, 1426], [890, 1063], [771, 1170], [755, 1114], [930, 1422],
            [697, 1142], [748, 1063], [827, 1189], [847, 1192], [794, 1185], [848, 1175],
            [713, 1132], [698, 1127], [736, 1120], [835, 1326], [907, 1527], [811, 1254],
            [656, 1075], [890, 1292], [860, 1504], [740, 1037], [867, 1207], [776, 1324],
            [699, 1104], [906, 1386], [814, 1144], [801, 1187], [822, 1177], [851, 1105],
            [825, 1213], [752, 1101], [765, 997], [799, 1146], [715, 963], [790, 1044],
            [850, 1182], [813, 1117], [736, 1098], [677, 1054], [995, 1359]],
            dtype=np.dtype(float)),
        'example': 'hawed',
        'unicode': 'ɔ',
    },
    'EH': {
        'data': np.array([
            [734, 1893], [740, 2186], [707, 2110], [693, 1991], [743, 1981], [712, 1881],
            [734, 2141], [732, 2189], [710, 2138], [704, 2023], [766, 1922], [673, 2389],
            [651, 2157], [817, 2242], [684, 1992], [756, 2008], [750, 1935], [706, 1986],
            [717, 2256], [727, 1871], [584, 2175], [762, 1902], [981, 2075], [794, 2046],
            [620, 2071], [741, 1997], [693, 2012], [597, 2426], [886, 2186], [588, 2130],
            [755, 1762], [885, 2322], [804, 1998], [718, 2005], [773, 2042], [694, 2160],
            [680, 2068], [694, 2052], [726, 2123], [714, 2255], [623, 1882], [718, 1952],
            [758, 2091], [676, 2196], [770, 1866], [683, 1851], [709, 2088], [817, 1984]],
            dtype=np.dtype(float)),
        'example': 'head',
        'unicode': 'ɛ',
    },
    'EI': {
        'data': np.array([
            [539, 2476], [477, 2704], [672, 2457], [601, 2577], [487, 2600], [492, 2437],
            [553, 2458], [496, 2665], [505, 2777], [455, 2433], [574, 2316], [502, 2674],
            [647, 2539], [596, 2478], [500, 2482], [617, 2697], [558, 2505], [572, 2428],
            [574, 2360], [498, 2218], [511, 2574], [513, 2435], [617, 2668], [610, 2598],
            [485, 2423], [578, 2474], [507, 2573], [504, 2501], [494, 2680], [487, 2408],
            [547, 2262], [500, 2889], [433, 2773], [591, 2562], [596, 2314], [500, 2622],
            [645, 2554], [434, 2568], [543, 2399], [555, 2766], [484, 2276], [442, 2464],
            [606, 2438], [485, 2619], [472, 2616], [546, 2237], [439, 2801], [664, 2455]],
            dtype=np.dtype(float)),
        'example': 'haid',
        'unicode': 'e',
    },
    'ER': {
        'data': np.array([
            [470, 1437], [500, 1441], [523, 1413], [619, 1716], [494, 1485], [527, 1636],
            [518, 1604], [516, 1983], [503, 1480], [495, 1596], [567, 1496], [501, 1555],
            [531, 1688], [506, 1604], [484, 1652], [597, 1588], [491, 1506], [524, 1547],
            [508, 1562], [469, 1499], [505, 1568], [508, 1687], [538, 1813], [617, 1565],
            [522, 1490], [507, 1749], [514, 1831], [580, 1407], [557, 1546], [555, 1491],
            [588, 1509], [549, 1749], [461, 1534], [537, 1509], [492, 1635], [455, 1769],
            [656, 1614], [509, 1365], [584, 1486], [491, 1655], [482, 1676], [487, 1434],
            [537, 1638], [500, 1637], [456, 1615], [604, 1441], [513, 1569], [503, 1766]],
            dtype=np.dtype(float)),
        'example': 'heard',
        'unicode': 'ɝ',
    },
    'IH': {
        'data': np.array([
            [496, 2260], [446, 2444], [486, 2332], [502, 2391], [497, 2379], [531, 2174],
            [535, 2380], [498, 2654], [500, 2272], [471, 2277], [453, 2427], [489, 2603],
            [555, 2216], [491, 2316], [485, 2129], [457, 2590], [474, 2362], [501, 2355],
            [449, 2173], [458, 2143], [440, 2394], [535, 2134], [516, 2567], [552, 2453],
            [455, 2304], [515, 2378], [504, 2334], [544, 2570], [501, 2449], [431, 2190],
            [478, 2157], [556, 2541], [486, 2454], [442, 2472], [486, 2253], [440, 2453],
            [477, 2554], [436, 2559], [501, 2292], [443, 2644], [489, 2175], [451, 2247],
            [442, 2316], [449, 2492], [444, 2237], [498, 2234], [463, 2603], [499, 2384]],
            dtype=np.dtype(float)),
        'example': 'hid',
        'unicode': 'ɪ',
    },
    'IY': {
        'data': np.array([
            [441, 2806], [435, 2890], [380, 2703], [428, 2767], [346, 2703], [439, 2676],
            [394, 2689], [493, 2967], [442, 2850], [477, 2866], [408, 2627], [435, 2868],
            [438, 2813], [440, 2706], [455, 2797], [443, 2944], [429, 2698], [492, 2695],
            [382, 2563], [434, 2512], [435, 2621], [433, 2634], [531, 2996], [467, 3009],
            [375, 2577], [497, 2795], [495, 2910], [508, 2731], [452, 2754], [461, 2359],
            [433, 2720], [505, 2982], [434, 2854], [426, 2775], [424, 2692], [380, 2816],
            [439, 2927], [430, 2606], [383, 2733], [438, 2964], [427, 2772], [437, 2555],
            [431, 2522], [331, 3049], [440, 2757], [489, 2641], [419, 2874], [437, 2778]],
            dtype=np.dtype(float)),
        'example': 'heed',
        'unicode': 'i',
    },
    'OA': {
        'data': np.array([
            [557, 1135], [432, 1013], [624, 1130], [627, 1314], [443, 879], [580, 995], [621, 1025],
            [506, 1257], [538, 998], [453, 980], [597, 1150], [688, 1185], [631, 1181], [636, 1041],
            [449, 924], [612, 1026], [616, 1034], [631, 1185], [504, 1047], [469, 934], [578, 1061],
            [560, 987], [508, 991], [658, 1045], [498, 970], [582, 946], [613, 1184], [636, 981],
            [636, 1096], [584, 936], [499, 1046], [504, 990], [453, 880], [641, 1195], [620, 1057],
            [563, 1198], [632, 1108], [430, 820], [438, 812], [450, 889], [444, 987], [435, 862],
            [608, 1007], [568, 1112], [624, 1109], [522, 923], [444, 803], [720, 1277]],
            dtype=np.dtype(float)),
        'example': 'boat',
        'unicode': 'o',
    },
    'OO': {
        'data': np.array([
            [489, 1357], [503, 1332], [494, 1102], [509, 1323], [506, 1188], [483, 987],
            [563, 1233], [546, 1534], [500, 1200], [491, 1179], [575, 1619], [502, 1322],
            [552, 1193], [499, 1103], [501, 1189], [562, 1109], [492, 1161], [505, 1404],
            [500, 1095], [496, 1169], [458, 1042], [617, 1373], [559, 1246], [509, 1244],
            [490, 1122], [616, 1331], [521, 1428], [570, 1117], [532, 1354], [482, 1124],
            [541, 1183], [615, 1375], [579, 1392], [552, 1364], [449, 1396], [464, 1169],
            [552, 1214], [486, 1088], [516, 1019], [444, 1081], [497, 1440], [499, 1144],
            [480, 1108], [520, 1421], [498, 1162], [519, 1051], [509, 1049], [553, 1135]],
            dtype=np.dtype(float)),
        'example': 'put',
        'unicode': 'ʊ',
    },
    'UH': {
        'data': np.array([
            [803, 1501], [788, 1547], [789, 1291], [731, 1552], [693, 1381], [795, 1362],
            [854, 1401], [716, 1634], [809, 1391], [680, 1499], [773, 1614], [817, 1604],
            [739, 1320], [742, 1352], [700, 1338], [824, 1459], [705, 1166], [839, 1555],
            [642, 1338], [701, 1318], [777, 1397], [686, 1477], [769, 1382], [815, 1427],
            [678, 1397], [823, 1419], [700, 1580], [885, 1580], [835, 1507], [756, 1531],
            [685, 1100], [828, 1454], [764, 1359], [729, 1435], [761, 1380], [809, 1468],
            [818, 1516], [710, 1293], [755, 1427], [827, 1416], [736, 1300], [678, 1229],
            [780, 1380], [693, 1277], [712, 1499], [733, 1378], [796, 1271], [811, 1450]],
            dtype=np.dtype(float)),
        'example': 'but',
        'unicode': 'ʌ',
    },
    'UW': {
        'data': np.array([
            [503, 1068], [435, 1384], [431, 788], [436, 1415], [455, 994], [437, 1113], [384, 1084],
            [503, 1514], [504, 1096], [473, 1113], [461, 1627], [477, 998], [445, 1136],
            [483, 1117], [466, 778], [440, 1122], [436, 911], [506, 1030], [360, 1034], [491, 1260],
            [391, 971], [431, 1003], [525, 1065], [448, 1369], [445, 881], [503, 963], [508, 1001],
            [510, 917], [502, 1050], [488, 1119], [477, 1184], [504, 1068], [483, 938], [448, 1463],
            [438, 1711], [437, 1178], [504, 1041], [429, 827], [438, 1078], [450, 916], [474, 974],
            [430, 954], [415, 1001], [362, 1351], [482, 987], [458, 1163], [467, 993], [491, 1317]],
            dtype=np.dtype(float)),
        'example': 'boot',
        'unicode': 'u',
    }
}


# _________________________________________________________________________________________________
# Data prep
# `````````````````````````````````````````````````````````````````````````````````````````````````
def full_hillenbrand_to_dict(filename=DEFAULT_HILLENBRAND):
    """ pulls and preps data for all 3 formants """
    csvdata = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=str)
    data_out = copy.copy(hillenbrand_data)
    for key in data_out.keys():
        data_out[key]['data'] = None

    for row in range(csvdata.shape[0]):
        key = csvdata[row, 0].upper()
        formants = np.array(csvdata[row, 1:], dtype=np.dtype(float))
        if not np.any(formants==0):
            if data_out[key]['data'] is None:
                data_out[key]['data'] = formants
            else:
                data_out[key]['data'] = np.vstack((data_out[key]['data'], formants))

    for i, key in enumerate(data_out.keys()):
        data_i = data_out[key]['data']
        if i == 0:
            data = data_i
            colors = np.array([i/12.0]*data_i.shape[0], dtype=float)
        else:
            data = np.vstack((data, data_i))
            colors = np.append(colors, np.array([i/12.0]*data_i.shape[0], dtype=float))

    return data_out


def gen_model(which_phonemes=None, n_per_phoneme=1, erb=False, f3=False):
    '''
    Generates a Teacher-ready model from the Hillenbrand data.

    Optional arguments:
        which_phonemes (list<str>): List of phonemes to include in the model. If None (default), all
            phonemes will be chosen.
        n_per_phoneme (int): number of datapoints per phoneme. Defaults to 1.
        erb (bool): If True, converts the Hillenbrand data from Hz to ERB. The resulting model will
            be in ERB space.
        f3 (bool): If True, generates data using the third formant as well. Not that beacuse there
            are 0 values in some of the F3 data that there will be fewer data point per phoneme.

    Returns:
        model (dict): model has two keys: "parameters" and "assignment". "parameters" is a list with
            an entry for each category. Each entry is a tuple that holds the component (likelihood)
            parameters for that component. "assignment" is a n-length numpy array which assigns
            data to components.
        which_phonems (list<str>): List of category names.

    Example (Three-phoneme model):
        >>> model, labels = gen_model(which_phonemes=['AW', 'IY', 'UW'], n_per_phoneme=2)
        >>> labels
        ['AW', 'IY', 'UW']
        >>> model['assignment']
        array([0, 0, 1, 1, 2, 2])
        >>> model['parameters'][0]
        (array([  801.0212766 ,  1188.27659574]), array([[  5172.15171138,   6057.42876966],
               [  6057.42876966,  16614.6827012 ]]))
        >>> model['parameters'][1]
        (array([  437.25  ,  2761.3125]), array([[  1650.06382979,   1277.85638298],
               [  1277.85638298,  21738.55984043]]))
        >>> model['parameters'][2]
        (array([  459.66666667,  1105.52083333]), array([[  1496.05673759,   -417.92907801],
               [  -417.92907801,  42130.33998227]]))
    '''
    if not which_phonemes:
        which_phonemes = all_the_vowels

    if not f3:
        input_data = hillenbrand_data
    else:
        input_data = full_hillenbrand_to_dict()

    parameters = []
    assignment = []
    for k, phoneme in enumerate(which_phonemes):
        formants = input_data[phoneme]['data']
        X = utils.hz_to_erb(formants) if erb else formants

        sigma = np.cov(X.T)
        mu = np.mean(X, axis=0)

        assignment += [k]*n_per_phoneme

        parameters.append((mu, sigma))

    model = {
        'parameters': parameters,
        'assignment': np.array(assignment, dtype=np.dtype(int)),
        'd': 2
    }

    if f3:
        model['d'] = 3

    return model, which_phonemes


# _________________________________________________________________________________________________
# IDS-specific plotting
# `````````````````````````````````````````````````````````````````````````````````````````````````
def plot_phoneme_models(teacher_data, target_model, labels, err_interval=.5, grayscale=False):
    """
    Plots the entire phoneme model. Creats scatter plots of the optimized and original data as well
    as their means and covariance matrices.

    Inputs:
        teacher_data (list<nump.ndarray>): a ids_teach.Teacher object's data atribute (teacher.data)
        target_model (dict): a ids_teach.Teacher object's target atrribute
        labels (list<string>): list of phone lables. these are used to lookup data in the
            hillenbrand data and must be sorted in the same order as the parameters in teh target
            model. If you use the values directy from gen_model, you will be fine.

    Kwargs:
        err_interval (float): The error interval to use for plotting covariance matricies.
        grayscale (bool): If True, plots in black and white (for publication)
    """
    fontsize = 14
    if grayscale:
        # FIXME: colors aren't so great
        color_opt = 'black'
        color_ads = 'white'
        font_color = 'gray'
    else:
        color_opt = 'blue'
        color_ads = 'red'
        font_color = 'white'

    # plot original data
    for i, phoneme in enumerate(labels):
        ads_data = hillenbrand_data[phoneme]['data']
        ads_mean = target_model['parameters'][i][0]
        ads_cov = target_model['parameters'][i][1]

        opt_data = teacher_data[i][:48, :]
        opt_mean = np.mean(teacher_data[i], axis=0)
        opt_cov = np.cov(teacher_data[i], rowvar=0)

        utils.plot_cov_ellipse(ads_cov, ads_mean, color=color_ads, ec=color_ads, lw=1, alpha=.2)
        utils.plot_cov_ellipse(opt_cov, opt_mean, color=color_opt, ec=color_opt, lw=1, alpha=.2)

        plt.scatter(ads_data[:, 0], ads_data[:, 1], color=color_ads)
        plt.scatter(ads_mean[0], ads_mean[1], color=color_ads, s=20**2)

        plt.scatter(opt_data[:, 0], opt_data[:, 1], color=color_opt)
        plt.scatter(opt_mean[0], opt_mean[1], color=color_opt, s=20**2)

        symbol = hillenbrand_data[phoneme]['unicode']
        kwargs = dict(fontsize=fontsize, color=font_color, ha='center', va='center')
        plt.text(ads_mean[0], ads_mean[1], symbol, **kwargs)
        plt.text(opt_mean[0], opt_mean[1], symbol, **kwargs)


def plot_phoneme_articulation(teacher_data, target_model, labels):
    """
    Plots the change in distance between each phoneme pair from ADS to optimized data. Sorts by
    distance. Black bars are corner vowel pairs.
    """
    num_phonemes = len(labels)

    # process ads_data
    means_ads = []
    phoneme_symbols = []
    for i, phoneme in enumerate(labels):
        means_ads.append(target_model['parameters'][i][0])
        phoneme_symbols.append(hillenbrand_data[phoneme]['unicode'])

    # prcess optimized data
    means_opt = []
    for data in teacher_data:
        means_phoneme = []
        for i in range(len(labels)):
            means_phoneme.append(np.mean(data[i], axis=0))
        means_opt.append(means_phoneme)

    dist_ads = []
    bar_colors = []
    pair_labels = []

    # generate labels and distance data
    deltas = np.zeros((len(teacher_data), nchoosek(num_phonemes, 2)))
    phoneme_pairs = it.combinations([i for i in range(num_phonemes)], 2)
    for j, (phoneme_1, phoneme_2) in enumerate(phoneme_pairs):
        pair_label = phoneme_symbols[phoneme_1] + '-' + phoneme_symbols[phoneme_2]
        pair_labels.append(pair_label)
        if labels[phoneme_1] in corner_vowels and labels[phoneme_2] in corner_vowels:
            bar_colors.append('black')
        else:
            bar_colors.append('white')

        dist_ads = utils.dist(means_ads[phoneme_1], means_ads[phoneme_2])
        for i in range(len(teacher_data)):
            dist_opt = utils.dist(means_opt[i][phoneme_1], means_opt[i][phoneme_2])
            delta = dist_opt-dist_ads
            # import pdb; pdb.set_trace()
            deltas[i, j] = delta

    delta_means = np.mean(deltas, axis=0)
    delta_stds = np.std(deltas, axis=0)

    indices = np.argsort(delta_means).tolist()
    delta_means = [delta_means[i] for i in indices]
    if len(teacher_data) > 1:
        delta_stds = [delta_stds[i] for i in indices]
    else:
        delta_stds = [0] * len(indices)
    pair_labels = [pair_labels[i] for i in indices]
    bar_colors = [bar_colors[i] for i in indices]

    num_phoneme_pairs = len(indices)
    bar_x = np.arange(num_phoneme_pairs)
    label_x = bar_x+0.5
    plt.bar(bar_x, delta_means, yerr=delta_stds, color=bar_colors)
    plt.plot([-.2, num_phoneme_pairs], [0, 0])
    plt.xlim([-.2, num_phoneme_pairs])
    plt.xticks(label_x, pair_labels, rotation='vertical')


def plot_phoneme_variation(teacher_data, target_model, labels):
    """
    Plots the change in F1 and F2 varaince and F1-F2 covariance for each phoneme from ADS to
    optimized data.
    """
    # FIXME: fill in
    pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
