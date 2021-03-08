from django.shortcuts import render
from django.http.response import HttpResponse

import csv
import pandas as pd

# Create your views here.


def index(request):
    # csv = request.FILES['data_source']
    # df = pd.read_csv(csv, encoding='big5')
    parameters_1 = [request.POST.get('id_1'), request.POST.get(
        'id_2'), request.POST.get('id_3')]
    parameters_2 = [request.POST.get('id_4'), request.POST.get(
        'id_5'), request.POST.get('id_6')]
    parameters_3 = [request.POST.get('id_7'), request.POST.get(
        'id_8'), request.POST.get('id_9')]
    context = {}

    return render(request, 'index.html', context)

# def check_col(request):
#     request['data_source'] = csv
#     pd.DataFrame.from_csv(csv)
#     return JsonResponse


def form_submit(request):
    if request.FILES:
        import pandas as pd
        import numpy as np
        csv = request.FILES['data_source']
        data = pd.read_csv(csv, encoding='utf-8')

        weight = [request.POST.get('id_1'), request.POST.get(
            'id_2'), request.POST.get('id_3'), request.POST.get('id_4')]

        myweight = np.array(weight, dtype=np.float32)

        criterion = [request.POST.get(
            'id_5'), request.POST.get('id_6'), request.POST.get('id_7'), request.POST.get('id_8')]

        mycriterion = np.array(criterion, dtype=np.float32)

        # Create an empty list
        Row_list = []
        Name_list = []
        my_list = []
        count_col = data.shape[1]  # 總共幾行(包括第一行的名稱)

        # Iterate over each row
        for index, rows in data.iterrows():
            # Create list for the current row
            if my_list == []:
                for j in range(1, count_col, 1):
                    my_list.append(rows[j])
            else:
                my_list = []
                for j in range(1, count_col, 1):
                    my_list.append(rows[j])
            #my_list =[rows[1], rows[2], rows[3], rows[4]]

            # append the list to the final list
            Row_list.append(my_list)
            Name_list.append(rows[0])

        C = None
        optimum_choice = None

        a = Row_list
        w = myweight
        I = mycriterion

        # Decision Matrix
        a = np.array(a, dtype=np.float).T
        assert len(a.shape) == 2, "Decision matrix a must be 2D"

        # Number of alternatives, aspects
        (n, J) = a.shape

        # Weight matrix
        w = np.array(w, dtype=np.float)
        assert len(w.shape) == 1, "Weights array must be 1D"
        assert w.size == n, "Weights array wrong length, " + \
            "should be of length {}".format(n)

        # Benefit (True) or Cost (False) criteria?
        I = np.array(I, dtype=np.int8)
        assert len(I.shape) == 1, "Criterion array must be 1D"
        assert len(I) == n, "Criterion array wrong length, " + \
            "should be of length {}".format(n)

        # Initialise best/worst alternatives lists
        ab, aw = np.zeros(n), np.zeros(n)

        # def step1():
        """ TOPSIS Step 1
        Calculate the normalised decision matrix (self.r)
        """
        r = a/np.array(np.linalg.norm(a, axis=1)[:, np.newaxis])

        # def step2(self):
        """ TOPSIS Step 2
        Calculate the weighted normalised decision matrix
        Two transposes required so that indices are multiplied correctly:
        """
        v = (w * r.T).T

        # def step3(self):
        """ TOPSIS Step 3
        Determine the ideal and negative-ideal solutions
        I[i] defines i as a member of the benefit criteria (True) or the cost
        criteria (False)
        """
        # Calcualte ideal/negative ideals
        ab = np.max(v, axis=1) * I + np.min(v, axis=1) * (1 - I)
        aw = np.max(v, axis=1) * (1 - I) + np.min(v, axis=1) * I

        # def step4(self):
        """ TOPSIS Step 4
        Calculate the separation measures, n-dimensional Euclidean distance
        """
        # Create two n long arrays containing Eculidean distances
        # Save the ideal and negative-ideal solutions
        db = np.linalg.norm(v - ab[:, np.newaxis], axis=0)
        dw = np.linalg.norm(v - aw[:, np.newaxis], axis=0)

        # def step5(self):
        """ TOPSIS Step 5 & 6
        Calculate the relative closeness to the ideal solution, then rank the
        preference order
        """
        # Ignore division by zero errors
        # np.seterr(all='ignore')
        # Find relative closeness
        C = dw / (dw + db)
        get_length = len(C)

        # --------好的-----------
        # 找好的
        optimum_choice = C.argsort()[-1]
        count_good = 0
        opt_idx = []
        for i in range(get_length):
            if C[i] == C[optimum_choice]:
                count_good += 1
                opt_idx.append(i)

        if count_good == 0:
            opt_idx.append(optimum_choice)

        # --------不好的-----------
        # 找不好的
        bad_choice = C.argsort()[0]
        count_bad = 0
        bad_idx = []
        for j in range(get_length):
            if C[j] == C[bad_choice]:
                count_bad += 1
                bad_idx.append(j)

        if count_bad == 0:
            bad_idx.append(bad_choice)

        # print('--------好的-----------')
        for idx in opt_idx:
            print('Name: {}(a[{}]) is : {}'.format(
                Name_list[idx], idx, a[:, idx]))
            result1 = 'Best choice: {}, {}'.format(Name_list[idx], a[:, idx])

        # print('--------不好的-----------')
        for idx in bad_idx:
            print('Name: {}(a[{}]) is : {}'.format(
                Name_list[idx], idx, a[:, idx]))
            result2 = 'Poor choice: {}, {}'.format(Name_list[idx], a[:, idx])

    context = {
        'csvData': data,
        'myweight': myweight,
        'mycriterion': mycriterion,
        'opt_idx': opt_idx,
        'bad_idx': bad_idx,
        'Name_list': Name_list,
        'a': a,
    }
    return render(request, 'result.html', context)