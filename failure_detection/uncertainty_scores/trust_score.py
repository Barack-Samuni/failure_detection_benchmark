import multiprocessing
import time

import numpy as np
from sklearn.neighbors import KDTree, KNeighborsClassifier


def get_trustscores(train_features, train_targets, test_features, test_predictions):
    trust_scorer = TrustScore(num_workers=4)
    trust_scorer.fit(
        train_features.numpy(),
        train_targets.int().numpy(),
    )
    test_trust_scores = trust_scorer.get_score(
        test_features.numpy(), test_predictions.int().numpy()
    )
    return {"TrustScore": test_trust_scores}


"""
Below: code taken from https://github.com/google/TrustScore


                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


class TrustScore:
    """
    Trust Score: a measure of classifier uncertainty based on nearest neighbors.
    """

    def __init__(
        self, k=10, alpha=0.0, filtering="none", min_dist=1e-12, num_workers=1
    ):
        """
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
        """
        self.k = k
        self.filtering = filtering
        self.alpha = alpha
        self.min_dist = min_dist
        self.num_workers = num_workers

    def filter_by_density(self, X):
        """Filter out points with low kNN density.
        Args:
        X: an array of sample points.
        Returns:
        A subset of the array without points in the bottom alpha-fraction of
        original points of kNN density.
        """
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        return X[np.where(knn_radii <= eps)[0], :]

    def filter_by_uncertainty(self, X, y):
        """Filter out points with high label disagreement amongst its kNN neighbors.
        Args:
        X: an array of sample points.
        Returns:
        A subset of the array without points in the bottom alpha-fraction of
        samples with highest disagreement amongst its k nearest neighbors.
        """
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        neigh.fit(X, y)
        confidence = neigh.predict_proba(X)
        cutoff = np.percentile(confidence, self.alpha * 100)
        unfiltered_idxs = np.where(confidence >= cutoff)[0]
        return X[unfiltered_idxs, :], y[unfiltered_idxs]

    def fit(self, X, y):
        """Initialize trust score precomputations with training data.
        WARNING: assumes that the labels are 0-indexed (i.e.
        0, 1,..., n_labels-1).
        Args:
        X: an array of sample points.
        y: corresponding labels.
        """
        self.n_labels = np.max(y) + 1
        self.kdtrees = [None] * self.n_labels
        if self.filtering == "uncertainty":
            X_filtered, y_filtered = self.filter_by_uncertainty(X, y)
        for label in range(self.n_labels):
            if self.filtering == "none":
                X_to_use = X[np.where(y == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "density":
                X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "uncertainty":
                X_to_use = X_filtered[np.where(y_filtered == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)

            if len(X_to_use) == 0:
                print(
                    "Filtered too much or missing examples from a label! Please lower alpha or check data."
                )

    def get_score(self, X, y_pred):
        """Compute the trust scores.
        Given a set of points, determines the distance to each class.
        Args:
        X: an array of sample points.
        y_pred: The predicted labels for these points.
        Returns:
        The trust score, which is ratio of distance to closest class that was not
        the predicted class to the distance to the predicted class.
        """
        d = np.tile(None, (X.shape[0], self.n_labels))

        if self.num_workers < 2:
            print("Using single processing for computing trust scores")
            for label_idx in range(self.n_labels):
                t0 = time.time()
                d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]
                print(
                    f"KDTree for label {label_idx} run in {time.time() - t0} secondes"
                )
        else:
            print(
                f"Enabling multiprocessing with {self.num_workers} workers for computing trust scores"
            )
            self.current_X = X
            pool = multiprocessing.Pool(self.num_workers)
            d = pool.map(self.process_d, range(self.n_labels))
            d = np.array(d).T

        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[range(d.shape[0]), y_pred]
        d_to_closest_not_pred = np.where(
            sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1]
        )
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)

    def process_d(self, label_idx):
        t0 = time.time()
        a = self.kdtrees[label_idx].query(self.current_X, k=2)[0][:, -1]
        print("KDTree for label %d run in %ls secondes" % (label_idx, time.time() - t0))
        return a

    class KNNConfidence:
        """Baseline which uses disagreement to kNN classifier."""

        def __init__(self, k=10):
            self.k = k

        def fit(self, X, y):
            self.kdtree = KDTree(X)
            self.y = y

        def get_score(self, X, y_pred):
            knn_idxs = self.kdtree.query(X, k=self.k)[1]
            knn_outputs = self.y[knn_idxs]
            return np.mean(
                knn_outputs == np.transpose(np.tile(y_pred, (self.k, 1))), axis=1
            )
