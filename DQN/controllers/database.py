import psycopg2
import re




class Database:
    database_name = 'imdbload'

    discard = "discard all;"

   # joins = [["movie_companies", "company_type"], ["movie_companies", "company_name"], ["movie_companies", "title"]
    #         ]

    #conditions = [

     #   "movie_companies.company_type_id=company_type.id", "movie_companies.company_id=company_name.id",
      #  "movie_companies.movie_id=title.id"

    #]

    #training_part_query = "SET default_tablespace = temp_tbs; SET join_collapse_limit = 1;"

    #optimizer_plan_query = "SET default_tablespace = temp_tbs; SET join_collapse_limit = 8; EXPLAIN ANALYSE   "

    joins = [["movie_companies", "company_type"], ["movie_companies", "company_name"], ["movie_companies", "title"],
             ["title", "kind_type"], ["movie_info", "title"]
             ]

    conditions = [

        "movie_companies.company_type_id=company_type.id", "movie_companies.company_id=company_name.id",
        "movie_companies.movie_id=title.id",
        "title.kind_id=kind_type.id", "movie_info.movie_id=title.id"]

    #database_name = 'shopx'

    """joins = [["customers", "orders"], ["customers", "reviews"], ["orders", "cashiers"],
             ["customers" , "redeemptions"]

             ]
    conditions = ["orders.customer_phone_number = customers.phone_number",
                  "customers.phone_number=reviews.phone_number", "orders.cashier_phone_number = cashiers.phone_number",
                  "customers.phone_number = redeemptions.phone_number"

                  ]"""

    query_explain_part = " EXPLAIN ANALYSE "
    query_select_part = " SELECT * FROM "
    query_force_plan_part = " SET default_tablespace = temp_tbs; SET join_collapse_limit = 1; "
    query_optimizer_select_plan = " SET default_tablespace = temp_tbs; SET join_collapse_limit = 8; "

    def __init__(self):
        connection = psycopg2.connect(user="postgres",
                                      password="2494",
                                      host="127.0.0.1",
                                      port="5432",
                                      database=self.database_name)
        self.cursor = connection.cursor()

    def execute_query(self,query):
        self.cursor.execute(query)

    def get_query_response_time(self,query):
        #self.execute_query(self.discard)
        self.execute_query(query)
        record = self.cursor.fetchall()
        data = record[len(record) - 1][0]
        return float(re.findall("\d+\.\d+", data)[0])

    def make_query(self, conditions_order,training=1):

        conditions_part = " WHERE " + self.conditions[conditions_order[0]]

        join_part = self.joins[conditions_order[0]][0] + " CROSS JOIN " + self.joins[conditions_order[0]][1]

        for i in range(len(conditions_order) - 1):
            conditions_part = conditions_part + " AND " + self.conditions[conditions_order[i + 1]]

            if join_part.find(self.joins[conditions_order[i + 1]][0]) == -1:
                join_part = join_part + " CROSS JOIN  " + self.joins[conditions_order[i + 1]][0]

            if join_part.find(self.joins[conditions_order[i + 1]][1]) == -1:
                join_part = join_part + " CROSS JOIN  " + self.joins[conditions_order[i + 1]][1]

        query = self.query_force_plan_part

        if training == 0:
            query = self.query_optimizer_select_plan

        return query + self.query_explain_part + self.query_select_part+ join_part + conditions_part+" LIMIT 1000000"
