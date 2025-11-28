import unittest
from collections import defaultdict
import uuid

from litaagent_std.inventory_manager_n import (
    InventoryManager,
    IMContract,
    IMContractType,
    MaterialType,
)


class TestInventoryManager(unittest.TestCase):

    def setUp(self):
        self.im = InventoryManager(
            raw_storage_cost=0.1,
            product_storage_cost=0.2,
            processing_cost=1.0,
            daily_production_capacity=10, # Default, can be overridden per test
            max_day=20
        )

    def _add_demand_contract(self, day, quantity, price=100, contract_id_prefix="d_test"):
        contract_id = f"{contract_id_prefix}_{day}_{quantity}_{price}_{uuid.uuid4().hex[:8]}"
        demand_contract = IMContract(
            contract_id=contract_id,
            partner_id="consumer1",
            type=IMContractType.DEMAND,
            quantity=float(quantity),
            price=float(price),
            delivery_time=int(day),
            bankruptcy_risk=0.0,
            material_type=MaterialType.PRODUCT
        )
        self.im.add_transaction(demand_contract)
        return demand_contract

    def _add_supply_contract(self, day, quantity, price=10, contract_id_prefix="s_test"):
        contract_id = f"{contract_id_prefix}_{day}_{quantity}_{price}_{uuid.uuid4().hex[:8]}"
        supply_contract = IMContract(
            contract_id=contract_id,
            partner_id="supplier1",
            type=IMContractType.SUPPLY,
            quantity=float(quantity),
            price=float(price),
            delivery_time=int(day),
            bankruptcy_risk=0.0,
            material_type=MaterialType.RAW
        )
        self.im.add_transaction(supply_contract)
        return supply_contract

    def test_initial_state_shortages_and_udpp(self):
        """Test initial shortages and UDPP are zero."""
        self.assertEqual(self.im.get_today_insufficient(0), 0, "Initial today's insufficient raw should be 0")
        self.assertEqual(self.im.get_total_insufficient(0), 0, "Initial total insufficient raw should be 0")
        udpp = self.im.get_udpp(0, self.im.max_day)
        self.assertEqual(sum(udpp.values()), 0, "Initial UDPP sum should be 0")

    def test_single_demand_contract_today(self):
        """Test UDPP and shortages with a single demand contract for today."""
        self._add_demand_contract(day=0, quantity=5)
        # add_transaction calls plan_production.
        # jit_production_plan_abs(start_day=0, inv_raw=0, inv_prod=0, future_prod_deliver={0:5})
        # -> prod_plan[0]=5, emer_raw[0]=5, planned_raw_demand[0]=5.
        # -> insufficient_raw[0]["daily"]=5, insufficient_raw[0]["total"]=5.

        udpp = self.im.get_udpp(0, self.im.max_day) # Calls plan_production again.
        # get_udpp: inv_on_hand_raw_summary=0. deliveries_raw={}.
        # Day 0: inv_on_hand_sim_raw=0. demand_today=prod_plan[0]=5. udpp[0]=5.
        self.assertEqual(udpp.get(0, 0), 5, "UDPP for day 0 should be 5")
        self.assertEqual(self.im.get_today_insufficient(0), 5,
                         "Today's insufficient raw for day 0 production should be 5")
        self.assertEqual(self.im.get_total_insufficient(0), 5, "Total insufficient raw including today should be 5")

    def test_single_demand_contract_future(self):
        """Test UDPP and shortages with a single demand contract for a future day."""
        self._add_demand_contract(day=5, quantity=8)
        # add_transaction calls plan_production.
        # jit_production_plan_abs(start_day=0, inv_raw=0, inv_prod=0, future_prod_deliver={5:8})
        # -> prod_plan[0-4]=0, prod_plan[5]=8.
        # -> emer_raw[0-4]=0, emer_raw[5]=8 (if no raw stock by then).
        # -> planned_raw_demand[0]=8 (total raw needed by day 0 for future demand at day 5).
        # -> insufficient_raw[0]["daily"]=0, insufficient_raw[0]["total"]=8.
        # -> insufficient_raw[5]["daily"]=8, insufficient_raw[5]["total"]=8.

        udpp_d0_view = self.im.get_udpp(0, self.im.max_day) # Calls plan_production again.
        # get_udpp(0,...): inv_on_hand_raw_summary=0. deliveries_raw={}. prod_plan from above.
        # Day 0-4: demand_today=0. udpp=0.
        # Day 5: inv_on_hand_sim_raw=0. demand_today=prod_plan[5]=8. udpp[5]=8.
        self.assertEqual(udpp_d0_view.get(5, 0), 8, "UDPP for day 5 should be 8 when viewed from day 0")
        self.assertEqual(sum(v for k, v in udpp_d0_view.items() if k != 5), 0, "UDPP for other days should be 0")

        self.assertEqual(self.im.get_today_insufficient(0), 0, "Today's (day 0) insufficient raw should be 0")
        self.assertEqual(self.im.get_total_insufficient(0), 8,
                         "Total insufficient raw (horizon from day 0) should include future demand")

        # Simulate moving to day 5
        for i in range(5):
            self.im.process_day_operations()
            self.im.update_day()

        self.assertEqual(self.im.current_day, 5)
        # At start of day 5, plan_production was last called at end of day 4 ops.
        # jit_production_plan_abs(start_day=4, inv_raw=0, inv_prod=0, future_prod_deliver={5:8})
        # -> prod_plan relative to day 4: prod_plan[4+1]=8. So, self.production_plan[5]=8.
        # -> insufficient_raw[5]["daily"]=8, insufficient_raw[5]["total"]=8.

        udpp_d5_view = self.im.get_udpp(5, self.im.max_day) # Calls plan_production(start_day=5)
        # get_udpp(5,...): inv_on_hand_raw_summary=0. deliveries_raw={}. prod_plan[5]=8.
        # Day 5: inv_on_hand_sim_raw=0. demand_today=prod_plan[5]=8. udpp[5]=8.
        self.assertEqual(udpp_d5_view.get(5, 0), 8, "UDPP for day 5 should be 8 when viewed from day 5")
        self.assertEqual(self.im.get_today_insufficient(5), 8, "Today's (day 5) insufficient raw should be 8")
        self.assertEqual(self.im.get_total_insufficient(5), 8, "Total insufficient (horizon from day 5) should be 8")

    def test_udpp_after_production_sufficient_raw(self):
        """Test UDPP reduction after production with sufficient raw materials."""
        self._add_supply_contract(day=0, quantity=10)
        self._add_demand_contract(day=0, quantity=7)

        # After contracts added, plan_production runs:
        # jit(start=0, inv_r=0, inv_p=0, fut_r={0:10}, fut_p={0:7})
        # -> raw_in[0]=10, dem[0]=7. M_d[0]=7. min_cum_prod=7. need_today=7.
        # -> raw_stock becomes 10. emer_raw[0]=max(0, 7-10)=0. prod_plan[0]=7.
        # -> insufficient_raw[0]["daily"]=0.
        # -> planned_raw_demand[0]: suf_prod=7, suf_raw_in=10. raw_after[0]=3. fut_sup=3. planned[0]=max(0,7-3)=4.
        # -> insufficient_raw[0]["total"]=4.
        self.assertEqual(self.im.get_today_insufficient(0), 0)
        self.assertEqual(self.im.get_total_insufficient(0), 4) # Total raw needed for future, considering current plan

        # get_udpp(0,...)
        # Day 0: inv_on_hand_sim_raw = 0 + 10 = 10. demand_today=prod_plan[0]=7. udpp[0]=max(0, 7-10)=0.
        self.assertEqual(self.im.get_udpp(0, self.im.max_day).get(0, 0), 0)

        product_stock_before_ops = self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock']
        result = self.im.process_day_operations()
        # process_day_operations:
        # 1. receive_materials: raw_batches gets 10. raw_inventory=10.
        # 2. execute_production(0): plan[0]=7. raw_stock=10. Produces 7. product_batches gets 7. raw_batches becomes 3.
        # 3. deliver_products(): demand=7. product_stock=7. Delivers 7. product_batches becomes 0.
        # 4. plan_production(): inv_raw=3, inv_prod=0. fut_p empty for day 0. prod_plan[0]=0.

        # Infer produced_today
        # Product stock after execute_production (before deliver_products) would be 7.
        # Product stock after deliver_products is 0.
        # delivered_products = 7.
        # produced_this_day = (current_product_stock_after_delivery(0) + delivered_products(7)) - product_stock_before_ops(0) = 7
        produced_this_day = (self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock'] + result['delivered_products']) - product_stock_before_ops
        self.assertEqual(produced_this_day, 7)
        self.assertEqual(result['delivered_products'], 7)

        udpp_after_prod = self.im.get_udpp(0, self.im.max_day) # plan_production just ran, prod_plan[0]=0
        # get_udpp(0,...): inv_raw_sum=3. deliveries_raw={}.
        # Day 0: inv_on_hand_sim_raw=3. demand_today=prod_plan[0]=0. udpp[0]=0.
        self.assertEqual(udpp_after_prod.get(0, 0), 0, "UDPP for day 0 should be 0 after production met demand")

        self.im.update_day() # current_day = 1
        # plan_production at end of day 0 ops: jit(start=1, inv_r=3, inv_p=0, ...). prod_plan all zeros. emer_raw all zeros.
        self.assertEqual(self.im.get_today_insufficient(1), 0)

    def test_udpp_after_production_insufficient_raw(self):
        """Test UDPP when production is limited by insufficient raw materials.
        IMN execute_production is all-or-nothing for the planned quantity.
        """
        self._add_supply_contract(day=0, quantity=3)
        demand_c_contract = self._add_demand_contract(day=0, quantity=7)

        # After contracts: plan_production(inv_raw=0, fut_r={0:3}, fut_p={0:7})
        # -> jit: raw_in[0]=3, dem[0]=7. M_d[0]=7. min_cum_prod=7. need_today=7.
        # -> raw_stock becomes 3. emer_raw[0]=max(0, 7-3)=4. prod_plan[0]=7.
        # -> insufficient_raw[0]["daily"]=4.
        # -> planned_raw_demand[0]: suf_prod=7, suf_raw_in=3. raw_after[0]=0. fut_sup=0. planned[0]=7.
        # -> insufficient_raw[0]["total"]=7.
        self.assertEqual(self.im.get_today_insufficient(0), 4)
        self.assertEqual(self.im.get_total_insufficient(0), 7)

        # get_udpp(0,...)
        # Day 0: inv_on_hand_sim_raw = 0 + 3 = 3. demand_today=prod_plan[0]=7. udpp[0]=max(0, 7-3)=4.
        self.assertEqual(self.im.get_udpp(0, self.im.max_day).get(0, 0), 4)

        product_stock_before_ops = self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock']
        result = self.im.process_day_operations()
        # process_day_operations:
        # 1. receive_materials: raw_batches gets 3. raw_inventory=3.
        # 2. execute_production(0): plan[0]=7. raw_stock=3. Raw insufficient (3 < 7).
        #    -> "原材料不足..." printed.
        #    -> insufficient_raw[0]["daily"] becomes max(4, 7)=7.
        #    -> insufficient_raw[0]["total"] becomes max(7, 7)=7.
        #    -> NO PRODUCTION.
        # 3. deliver_products(): demand_contract for 7 units. total_available_product=0.
        #    -> shortfall = 7.
        #    -> insufficient_raw[0]["daily"] becomes max(7,7)=7 (product shortfall).
        #    -> contract for 7 units (d0_1) remains in _pending_demand.
        # 4. plan_production(): inv_raw=3, inv_prod=0. fut_p={0:7}.
        #    -> jit(start=0, inv_r=3, inv_p=0, fut_r={}, fut_p={0:7})
        #    -> prod_plan[0]=7. emer_raw[0]=4. planned_raw[0]=7.

        produced_this_day = (self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock'] + result['delivered_products']) - product_stock_before_ops
        self.assertEqual(produced_this_day, 0) # No production occurred
        self.assertEqual(result['delivered_products'], 0) # No products to deliver

        udpp_after_prod = self.im.get_udpp(0, self.im.max_day) # plan_production just ran, prod_plan[0]=7
        # get_udpp(0,...): inv_raw_sum=3. deliveries_raw={}.
        # Day 0: inv_on_hand_sim_raw=3. demand_today=prod_plan[0]=7. udpp[0]=4.
        self.assertEqual(udpp_after_prod.get(0, 0), 4)

        # Check product shortfall recorded by deliver_products
        # This is a product shortfall, not raw material for production.
        # The key 'daily' for insufficient_raw gets updated by deliver_products for product shortfall.
        self.assertEqual(self.im.insufficient_raw.get(0, {}).get("daily", 0), 7)


    def test_udpp_production_limited_by_capacity_as_per_imn(self):
        """Test UDPP when production plan might exceed capacity,
        but execute_production follows the plan if raw materials are sufficient.
        IMN's daily_production_capacity influences planning, not a hard execution cap.
        """
        self.im.daily_production_capacity = 5 # This affects cap in jit_production_plan_abs
        self._add_supply_contract(day=0, quantity=10)
        demand_c = self._add_demand_contract(day=0, quantity=8)

        # After contracts, plan_production:
        # jit(start=0, cap=5, inv_r=0, inv_p=0, fut_r={0:10}, fut_p={0:8})
        # -> raw_in[0]=10, dem[0]=8. cap=5.
        # -> M_d[0] = dem[0] - cap*0 = 8.
        # -> min_cum_prod = max(0, M_d[0] + cap*0 - 0) = 8.
        # -> need_today = 8. prod_plan[0]=8.
        # -> raw_stock becomes 10. emer_raw[0]=max(0, 8-10)=0.
        # -> insufficient_raw[0]["daily"]=0.
        # -> planned_raw_demand[0]: suf_prod=8, suf_raw_in=10. raw_after[0]=2. fut_sup=2. planned[0]=max(0,8-2)=6.
        # -> insufficient_raw[0]["total"]=6.
        self.assertEqual(self.im.get_today_insufficient(0), 0)
        self.assertEqual(self.im.get_total_insufficient(0), 6)

        # get_udpp(0,...)
        # Day 0: inv_on_hand_sim_raw = 0 + 10 = 10. demand_today=prod_plan[0]=8. udpp[0]=max(0, 8-10)=0.
        self.assertEqual(self.im.get_udpp(0, self.im.max_day).get(0, 0), 0)

        product_stock_before_ops = self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock']
        result = self.im.process_day_operations()
        # process_day_operations:
        # 1. receive_materials: raw_batches gets 10.
        # 2. execute_production(0): plan[0]=8. raw_stock=10. Produces 8. (IMN execute_production does not cap by self.daily_production_capacity)
        # 3. deliver_products(): demand=8. product_stock=8. Delivers 8.
        # 4. plan_production(): inv_raw=2, inv_prod=0. fut_p empty. prod_plan[0]=0.

        produced_this_day = (self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock'] + result['delivered_products']) - product_stock_before_ops
        self.assertEqual(produced_this_day, 8)
        self.assertEqual(result['delivered_products'], 8)

        udpp_after_prod = self.im.get_udpp(0, self.im.max_day)
        # get_udpp(0,...): inv_raw_sum=2. deliveries_raw={}.
        # Day 0: inv_on_hand_sim_raw=2. demand_today=prod_plan[0]=0. udpp[0]=0.
        self.assertEqual(udpp_after_prod.get(0, 0), 0)
        self.assertEqual(self.im.insufficient_raw.get(0, {}).get("daily", 0), 0) # No product shortfall

    def test_multiple_demands_and_supplies_complex_imn_behavior(self):
        self.im.daily_production_capacity = 7 # cap=7

        # Day 0
        d0_1 = self._add_demand_contract(day=0, quantity=5, contract_id_prefix="d0_1")
        self._add_supply_contract(day=0, quantity=3, contract_id_prefix="s0_1")
        # plan_production after both: jit(start=0, cap=7, inv_r=0, inv_p=0, fut_r={0:3}, fut_p={0:5})
        # -> raw_in[0]=3, dem[0]=5. M_d[0]=5. min_cum_prod=5. need_today=5.
        # -> raw_stock becomes 3. emer_raw[0]=max(0, 5-3)=2. prod_plan[0]=5.
        # -> insufficient_raw[0]["daily"]=2.
        # -> planned_raw_demand[0]: suf_prod=5, suf_raw_in=3. raw_after[0]=0. fut_sup=0. planned[0]=5.
        # -> insufficient_raw[0]["total"]=5.
        self.assertEqual(self.im.get_udpp(0, self.im.max_day).get(0, 0), 2) # UDPP = need - available_today_supply = 5-3=2
        self.assertEqual(self.im.get_today_insufficient(0), 2) # Emergency raw needed

        raw_stock_before_ops_d0 = self.im.get_inventory_summary(0, MaterialType.RAW)['current_stock']
        product_stock_before_ops_d0 = self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock']
        result_d0 = self.im.process_day_operations()
        # 1. receive_materials: raw_batches gets 3. raw_inventory=3.
        # 2. execute_production(0): plan[0]=5. raw_stock=3. Raw insufficient (3 < 5).
        #    -> "原材料不足..." printed. insufficient_raw[0]["daily"]=max(2,5)=5. No production.
        # 3. deliver_products(): demand d0_1 (qty 5). total_available_product=0. Delivers 0. Shortfall 5.
        #    -> insufficient_raw[0]["daily"]=max(5,5)=5 (product shortfall). d0_1 remains pending.
        # 4. plan_production(): inv_raw=3, inv_prod=0. fut_p={0:5}. prod_plan[0]=5. emer[0]=2.
        self.assertEqual(self.im.get_inventory_summary(0, MaterialType.RAW)['current_stock'], 3.0) # Raw material received
        produced_d0 = (self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock'] + result_d0['delivered_products']) - product_stock_before_ops_d0
        self.assertEqual(produced_d0, 0)
        self.assertEqual(result_d0['delivered_products'], 0)
        self.assertEqual(self.im.insufficient_raw.get(0, {}).get("daily", 0), 5) # Product shortfall for d0_1
        self.im.update_day() # current_day = 1

        # Day 1
        d1_1 = self._add_demand_contract(day=1, quantity=6, contract_id_prefix="d1_1")
        self._add_supply_contract(day=1, quantity=10, contract_id_prefix="s1_1")
        # plan_production: jit(start=1, cap=7, inv_r=3, inv_p=0, fut_r={1:10}, fut_p={0:5 (past), 1:6})
        # Past demand d0_1 (qty 5) is ignored by jit_production_plan_abs as start_day=1.
        # -> raw_in[0]=10 (relative to day 1), dem[0]=6 (relative to day 1).
        # -> M_d[0]=6 (relative). min_cum_prod=6. need_today=6.
        # -> raw_stock (day 1, initial for jit) = 3. raw_stock (after raw_in[0]) = 3+10=13.
        # -> emer_raw[1]=max(0, 6-13)=0. prod_plan[1]=6.
        # -> insufficient_raw[1]["daily"]=0.
        # -> planned_raw_demand[1]: suf_prod=6, suf_raw_in=10. raw_after[0]=7. fut_sup=7. planned[1]=max(0,6-7)=0.
        # -> insufficient_raw[1]["total"]=0.
        udpp_d1_view = self.im.get_udpp(1, self.im.max_day)
        self.assertEqual(udpp_d1_view.get(1, 0), 0) # UDPP = need - available_today_supply = 6 - (3+10) = 0
        self.assertEqual(self.im.get_today_insufficient(1), 0)

        raw_stock_before_ops_d1 = self.im.get_inventory_summary(1, MaterialType.RAW)['current_stock'] # Should be 3
        product_stock_before_ops_d1 = self.im.get_inventory_summary(1, MaterialType.PRODUCT)['current_stock'] # Should be 0
        result_d1 = self.im.process_day_operations()
        # 1. receive_materials: raw_batches gets 10. raw_inventory=3+10=13.
        # 2. execute_production(1): plan[1]=6. raw_stock=13. Produces 6. product_batches gets 6. raw_inventory=7.
        # 3. deliver_products(): demand d1_1 (qty 6). total_available_product=6. Delivers 6. d1_1 satisfied.
        # 4. plan_production(): inv_raw=7, inv_prod=0. fut_p={0:5 (past)}. prod_plan will be zeros from day 1.
        self.assertAlmostEqual(self.im.get_inventory_summary(1, MaterialType.RAW)['current_stock'], 7.0)
        produced_d1 = (self.im.get_inventory_summary(1, MaterialType.PRODUCT)['current_stock'] + result_d1['delivered_products']) - product_stock_before_ops_d1
        self.assertEqual(produced_d1, 6)
        self.assertEqual(result_d1['delivered_products'], 6)
        self.assertEqual(self.im.insufficient_raw.get(1, {}).get("daily", 0), 0) # No product shortfall for d1_1
        self.im.update_day() # current_day = 2

        # Day 2
        d2_1 = self._add_demand_contract(day=2, quantity=12, contract_id_prefix="d2_1")
        # plan_production: jit(start=2, cap=7, inv_r=7, inv_p=0, fut_r={}, fut_p={0:5 (past), 2:12})
        # -> dem[0]=12 (relative to day 2). M_d[0]=12. min_cum_prod=12. need_today=12.
        # -> raw_stock (day 2, initial for jit) = 7.
        # -> emer_raw[2]=max(0, 12-7)=5. prod_plan[2]=12.
        # -> insufficient_raw[2]["daily"]=5.
        # -> planned_raw_demand[2]: suf_prod=12, suf_raw_in=0. raw_after[0]=0. fut_sup=0. planned[2]=12.
        # -> insufficient_raw[2]["total"]=12.
        udpp_d2_view = self.im.get_udpp(2, self.im.max_day)
        self.assertEqual(udpp_d2_view.get(2, 0), 5) # UDPP = need - available_today_supply = 12 - 7 = 5
        self.assertEqual(self.im.get_today_insufficient(2), 5)

        raw_stock_before_ops_d2 = self.im.get_inventory_summary(2, MaterialType.RAW)['current_stock'] # Should be 7
        product_stock_before_ops_d2 = self.im.get_inventory_summary(2, MaterialType.PRODUCT)['current_stock'] # Should be 0
        result_d2 = self.im.process_day_operations()
        # 1. receive_materials: no new raw. raw_inventory=7.
        # 2. execute_production(2): plan[2]=12. raw_stock=7. Raw insufficient (7 < 12).
        #    -> "原材料不足..." printed. insufficient_raw[2]["daily"]=max(5,12)=12. No production.
        # 3. deliver_products(): demand d2_1 (qty 12). total_available_product=0. Delivers 0. Shortfall 12.
        #    -> insufficient_raw[2]["daily"]=max(12,12)=12 (product shortfall). d2_1 remains pending.
        # 4. plan_production(): inv_raw=7, inv_prod=0. fut_p={0:5 (past), 2:12}. prod_plan[2]=12. emer[2]=5.
        self.assertAlmostEqual(self.im.get_inventory_summary(2, MaterialType.RAW)['current_stock'], 7.0)
        produced_d2 = (self.im.get_inventory_summary(2, MaterialType.PRODUCT)['current_stock'] + result_d2['delivered_products']) - product_stock_before_ops_d2
        self.assertEqual(produced_d2, 0)
        self.assertEqual(result_d2['delivered_products'], 0)
        self.assertEqual(self.im.insufficient_raw.get(2, {}).get("daily", 0), 12) # Product shortfall for d2_1
        self.im.update_day()

        # Check persistent product shortfall for d0_1
        # d0_1 (qty 5) was never satisfied. deliver_products on day 0 recorded a product shortfall of 5.
        # This value should persist in insufficient_raw[0]["daily"] if not overwritten by other logic.
        # However, insufficient_raw is typically for raw materials. The use by deliver_products for product shortfall is a bit of an overload.
        # Let's check if d0_1 is still pending.
        pending_d0_contracts = [c for c in self.im.get_pending_contracts(is_supply=False) if c.contract_id == d0_1.contract_id]
        self.assertEqual(len(pending_d0_contracts), 1, "d0_1 should still be pending")
        if pending_d0_contracts:
            self.assertEqual(pending_d0_contracts[0].quantity, 5, "Unsatisfied quantity for d0_1 should be 5")


    def test_udpp_clears_for_past_days_when_viewed_from_future(self):
        self._add_demand_contract(day=0, quantity=5)
        # plan_production: prod_plan[0]=5. emer_raw[0]=5.
        udpp_day0_view = self.im.get_udpp(0, self.im.max_day)
        # get_udpp(0,...): inv_raw_sim=0. demand_today=5. udpp[0]=5.
        self.assertEqual(udpp_day0_view.get(0, 0), 5)

        self.im.process_day_operations() # No raw, no prod, no delivery.
        # plan_production at end of day 0: inv_raw=0, inv_prod=0. fut_p={0:5}. prod_plan[0]=5.
        self.im.update_day()  # current_day = 1

        udpp_day1_view = self.im.get_udpp(1, self.im.max_day)
        # plan_production(start_day=1): fut_p={0:5} is past, ignored. prod_plan will be all zeros from day 1.
        # get_udpp(1,...): inv_raw_sim=0. demand_today=0 for all days >=1. udpp all zeros.
        self.assertEqual(udpp_day1_view.get(0, 0), 0, "UDPP for past day 0 should be 0 when viewed from day 1")
        self.assertEqual(sum(udpp_day1_view.values()), 0)


    def test_get_today_insufficient_with_supply_and_demand(self):
        self.im.current_day = 0
        self._add_demand_contract(day=0, quantity=8)
        # plan_production: jit(start=0, inv_r=0, inv_p=0, fut_p={0:8})
        # -> prod_plan[0]=8. emer_raw[0]=8. insufficient_raw[0]["daily"]=8.
        self._add_supply_contract(day=0, quantity=2)
        # plan_production: jit(start=0, inv_r=0, inv_p=0, fut_r={0:2}, fut_p={0:8})
        # -> raw_in[0]=2, dem[0]=8. M_d[0]=8. min_cum_prod=8. need_today=8.
        # -> raw_stock becomes 2. emer_raw[0]=max(0, 8-2)=6. prod_plan[0]=8.
        # -> insufficient_raw[0]["daily"]=6.
        self.assertEqual(self.im.get_today_insufficient(0), 6) # Should be 6, not 8

        raw_stock_before_ops = self.im.get_inventory_summary(0, MaterialType.RAW)['current_stock'] # 0
        self.im.process_day_operations()
        # 1. receive_materials: raw_batches gets 2. raw_inventory=2.
        # 2. execute_production(0): plan[0]=8. raw_stock=2. Raw insufficient (2 < 8).
        #    -> "原材料不足..." printed. insufficient_raw[0]["daily"]=max(6,8)=8. No production.
        # 3. deliver_products(): demand for 8. product_available=0. Delivers 0. Shortfall 8.
        #    -> insufficient_raw[0]["daily"]=max(8,8)=8 (product shortfall).
        # 4. plan_production(): inv_raw=2, inv_prod=0. fut_p={0:8}. prod_plan[0]=8. emer[0]=6.
        self.assertEqual(self.im.get_inventory_summary(0, MaterialType.RAW)['current_stock'], 2.0) # Raw material is kept
        self.im.update_day()


    def test_udpp_multiple_days_and_capacity_imn_behavior(self):
        self.im.daily_production_capacity = 5 # cap=5
        self._add_demand_contract(day=0, quantity=7)
        self._add_demand_contract(day=1, quantity=6)
        self._add_supply_contract(day=0, quantity=20)

        # plan_production: jit(start=0, cap=5, inv_r=0, inv_p=0, fut_r={0:20}, fut_p={0:7, 1:6})
        # -> raw_in[0]=20. dem=[7,6]. cap=5.
        # -> cum_dem=[7,13].
        # -> M_d[1]=13-5*1=8. M_d[0]=max(8, 7-5*0)=8.
        # Day 0: min_cum=max(0,M_d[0]+0-0)=8. need=8. raw_stock=0+20=20. emer=0. prod_plan[0]=8. raw_after[0]=12.
        # Day 1: raw_stock=12+0=12. min_cum=max(0,M_d[1]+5*1-0)=max(0,8+5)=13. need=max(0,13-8)=5. emer=0. prod_plan[1]=5. raw_after[1]=7.
        # -> prod_plan={0:8, 1:5}. insufficient_raw[0]["daily"]=0, insufficient_raw[1]["daily"]=0.

        # get_udpp(0,...)
        # Day 0: inv_on_hand_sim_raw = 0 + 20 = 20. demand_today=prod_plan[0]=8. udpp[0]=0.
        self.assertEqual(self.im.get_udpp(0, self.im.max_day).get(0, 0), 0)
        self.assertEqual(self.im.get_today_insufficient(0), 0)

        product_stock_before_ops_d0 = self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock']
        result_d0 = self.im.process_day_operations()
        # 1. receive_materials: raw_batches gets 20.
        # 2. execute_production(0): plan[0]=8. raw_stock=20. Produces 8. raw_inventory=12.
        # 3. deliver_products(): demand=7. product_stock=8. Delivers 7. product_inventory=1.
        # 4. plan_production(): inv_raw=12, inv_prod=1. fut_p={1:6}.
        #    jit(start=0, cap=5, inv_r=12, inv_p=1, fut_r={}, fut_p={1:6}) (assuming current_day is still 0 for this plan)
        #    dem relative to day 0: dem[1]=6.
        #    M_d[1]=6-5*1=1. M_d[0]=max(1,0-0)=1.
        #    Day 0: min_cum=max(0,1-1)=0. need=0. prod_plan[0]=0.
        #    Day 1: raw_stock=12. min_cum=max(0,1+5*1-1)=5. need=max(0,5-0)=5. prod_plan[1]=5.
        produced_d0 = (self.im.get_inventory_summary(0, MaterialType.PRODUCT)['current_stock'] + result_d0['delivered_products']) - product_stock_before_ops_d0
        self.assertEqual(produced_d0, 8)
        self.assertEqual(self.im.get_inventory_summary(0, MaterialType.RAW)['current_stock'], 12.0)
        self.im.update_day()  # current_day = 1

        # Day 1
        # At start of day 1, plan_production was called at end of day 0.
        # prod_plan={0:0, 1:5}. insufficient_raw[1]["daily"]=0.
        # get_udpp(1,...)
        # Day 1: inv_raw_sum=12. deliveries_raw={}. inv_on_hand_sim_raw=12. demand_today=prod_plan[1]=5. udpp[1]=0.
        self.assertEqual(self.im.get_udpp(1, self.im.max_day).get(1, 0), 0)
        self.assertEqual(self.im.get_today_insufficient(1), 0)

        product_stock_before_ops_d1 = self.im.get_inventory_summary(1, MaterialType.PRODUCT)['current_stock'] # Should be 1
        result_d1 = self.im.process_day_operations()
        # 1. receive_materials: none.
        # 2. execute_production(1): plan[1]=5. raw_stock=12. Produces 5. raw_inventory=7. product_inventory=1+5=6.
        # 3. deliver_products(): demand=6. product_stock=6. Delivers 6. product_inventory=0.
        # 4. plan_production(): inv_raw=7, inv_prod=0. fut_p empty. prod_plan all zeros.
        produced_d1 = (self.im.get_inventory_summary(1, MaterialType.PRODUCT)['current_stock'] + result_d1['delivered_products']) - product_stock_before_ops_d1
        self.assertEqual(produced_d1, 5)
        self.assertEqual(self.im.get_inventory_summary(1, MaterialType.RAW)['current_stock'], 7.0)
        self.im.update_day()  # current_day = 2

    def test_get_total_insufficient_with_supply_offsetting_future_demand_imn(self):
        # Demands: d0=5, d1=4, d2=3. Supplies: s0=2, s1=1. cap=10.
        self._add_demand_contract(day=0, quantity=5)
        self._add_demand_contract(day=1, quantity=4)
        self._add_demand_contract(day=2, quantity=3)
        self._add_supply_contract(day=0, quantity=2)
        self._add_supply_contract(day=1, quantity=1)

        # After all contracts, plan_production runs:
        # jit(start=0, cap=10, inv_r=0, inv_p=0, fut_r={0:2, 1:1}, fut_p={0:5, 1:4, 2:3})
        # -> raw_in=[2,1,0]. dem=[5,4,3].
        # -> prod_plan = {0:5, 1:4, 2:3}.
        # -> emer_raw = {0:3, 1:3, 2:3}.
        # -> planned_raw_demand = {0:11, 1:7, 2:3}.
        # -> insufficient_raw = {0:{"d":3,"t":11}, 1:{"d":3,"t":7}, 2:{"d":3,"t":3}}

        # UDPP values
        udpp = self.im.get_udpp(0, self.im.max_day)
        # Day 0: inv_sim_raw=0+2=2. demand=5. udpp[0]=3. inv_sim_raw becomes 0.
        # Day 1: inv_sim_raw=0+1=1. demand=4. udpp[1]=3. inv_sim_raw becomes 0.
        # Day 2: inv_sim_raw=0+0=0. demand=3. udpp[2]=3. inv_sim_raw becomes 0.
        self.assertEqual(udpp.get(0, 0), 3)
        self.assertEqual(udpp.get(1, 0), 3)
        self.assertEqual(udpp.get(2, 0), 3)

        self.assertEqual(self.im.get_total_insufficient(0), 11)
        self.assertEqual(self.im.get_total_insufficient(1), 7)
        self.assertEqual(self.im.get_total_insufficient(2), 3)

        # Process day 0
        self.im.process_day_operations()
        # 1. receive: raw=2.
        # 2. execute_production(0): plan[0]=5. raw=2. Insufficient. No production.
        #    insufficient_raw[0]["daily"]=max(3,5)=5.
        # 3. deliver_products(0): demand=5. prod_avail=0. Delivers 0. Shortfall 5.
        #    insufficient_raw[0]["daily"]=max(5,5)=5 (product shortfall).
        # 4. plan_production: inv_r=2, inv_p=0. fut_r={1:1}, fut_p={0:5,1:4,2:3}
        #    jit(start=0, inv_r=2, inv_p=0, fut_r={1:1}, fut_p={0:5,1:4,2:3})
        #    -> prod_plan={0:5,1:4,2:3}. emer={0:3,1:3,2:3}. planned={0:11,1:7,2:3}
        self.im.update_day()  # current_day = 1. Raw inventory is 2.

        # get_total_insufficient(1)
        # plan_production at end of day 0 ops:
        # jit(start=1, inv_r=2, inv_p=0, fut_r={1:1}, fut_p={0:5(past),1:4,2:3})
        # -> raw_in=[1,0] (relative to day 1). dem=[4,3] (relative to day 1).
        # -> prod_plan relative to day 1: {1:4, 2:3}.
        # -> emer_raw relative to day 1: {1:1, 2:3} (since inv_r=2, raw_in[0]=1 -> total 3 for day1 prod of 4)
        # -> planned_raw_demand relative to day 1: {1:7, 2:3}
        # So, self.insufficient_raw[1]["total"] = 7.
        self.assertEqual(self.im.get_total_insufficient(1), 7)


if __name__ == '__main__':
    unittest.main()
